from functools import partial
from typing import Callable, Optional

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tree_utils
import x_xy
from x_xy import maths
from x_xy.utils import distribute_batchsize, expand_batchsize

from neural_networks.logging import Logger
from neural_networks.rnno.optimizer import adam
from neural_networks.rnno.training_loop import TrainingLoop, TrainingLoopCallback
from neural_networks.rnno.training_loop_callbacks import (
    EvalFnCallback,
    _build_eval_fn,
    _repeat_state,
)

default_metrices = {
    "mae_deg": (
        lambda q, qhat: maths.angle_error(q, qhat),
        lambda arr: jnp.rad2deg(jnp.mean(arr, axis=(0, 1))),
    ),
    "rmse_deg": (
        lambda q, qhat: maths.angle_error(q, qhat) ** 2,
        # we reduce time at axis=1, and batchsize at axis=0
        lambda arr: jnp.rad2deg(jnp.mean(jnp.sqrt(jnp.mean(arr, axis=1)), axis=0)),
    ),
    "q90_ae_deg": (
        lambda q, qhat: maths.angle_error(q, qhat),
        lambda arr: jnp.rad2deg(jnp.mean(jnp.quantile(arr, 0.90, axis=1), axis=0)),
    ),
    "q99_ae_deg": (
        lambda q, qhat: maths.angle_error(q, qhat),
        lambda arr: jnp.rad2deg(jnp.mean(jnp.quantile(arr, 0.99, axis=1), axis=0)),
    ),
}


default_loss_fn = lambda q, qhat: maths.angle_error(q, qhat) ** 2


def _build_step_fn(
    metric_fn,
    apply_fn,
    initial_state,
    pmap_size,
    vmap_size,
    optimizer,
    tbp,
):
    """Build step function that optimizes filter parameters based on `metric_fn`.
    `initial_state` has shape (pmap, vmap, state_dim)"""

    @partial(jax.value_and_grad, has_aux=True)
    def loss_fn(params, state, X, y):
        yhat, state = jax.vmap(apply_fn, in_axes=(None, 0, 0))(params, state, X)
        pipe = lambda q, qhat: jnp.mean(jax.vmap(jax.vmap(metric_fn))(q, qhat))
        error_tree = jax.tree_map(pipe, y, yhat)
        return jnp.mean(tree_utils.batch_concat(error_tree, 0)), state

    @partial(
        jax.pmap,
        in_axes=(None, 0, 0, 0),
        out_axes=((None, 0), None),
        axis_name="devices",
    )
    def pmapped_loss_fn(params, state, X, y):
        pmean = lambda arr: jax.lax.pmean(arr, axis_name="devices")
        if isinstance(params, optax.LookaheadParams):
            fast_params = params.fast
        else:
            fast_params = params
        (loss, state), grads = loss_fn(fast_params, state, X, y)
        return (pmean(loss), state), pmean(grads)

    @jax.jit
    def apply_grads(grads, params, opt_state):
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state

    def step_fn(params, opt_state, X, y):
        N = tree_utils.tree_shape(X, axis=-2)
        X, y = expand_batchsize((X, y), pmap_size, vmap_size)
        nonlocal initial_state

        debug_grads = []
        state = initial_state
        for X_tbp, y_tbp in tree_utils.tree_split((X, y), int(N / tbp), axis=-2):
            (loss, state), grads = pmapped_loss_fn(params, state, X_tbp, y_tbp)
            debug_grads.append(grads)
            state = jax.lax.stop_gradient(state)
            params, opt_state = apply_grads(grads, params, opt_state)

        return params, opt_state, {"loss": loss}, debug_grads

    return step_fn


key_generator, key_network = jax.random.split(jax.random.PRNGKey(0))


def train(
    generator: Callable,
    n_episodes: int,
    network: hk.TransformedWithState,
    optimizer=adam(),
    tbp: int = 1000,
    loggers: list[Logger] = [],
    callbacks: list[TrainingLoopCallback] = [],
    initial_params: Optional[dict] = None,
    key_network: jax.random.PRNGKey = key_network,
    key_generator: jax.random.PRNGKey = key_generator,
    optimizer_uses_lookahead: bool = True,
):
    """Trains RNNO

    Args:
        generator (Callable): output of the rcmg-module
        n_episodes (int): number of episodes to train for
        network (hk.TransformedWithState): RNNO network
        optimizer (_type_, optional): optimizer, see optimizer.py module
        tbp (int, optional): Truncated backpropagation through time step size
        loggers: list of Loggers used to log the training progress.
        callbacks: callbacks of the TrainingLoop.
        initial_params: If given uses as initial parameters.
        key_network: PRNG Key that inits the network state and parameters.
        key_generator: PRNG Key that inits the data stream of the generator.
    """

    # test if generator is batched..
    key = jax.random.PRNGKey(0)
    X, y = generator(key)

    if tree_utils.tree_ndim(X) == 2:
        # .. if not then batch it
        generator = x_xy.algorithms.batch_generator(generator, 1)

    # .. now it most certainly is; Queue it for data
    X, y = generator(key)

    batchsize = tree_utils.tree_shape(X)
    pmap_size, vmap_size = distribute_batchsize(batchsize)

    params, initial_state = network.init(
        key_network,
        tree_utils.tree_slice(X, 0),
    )
    initial_state = _repeat_state(initial_state, batchsize)

    if initial_params is not None:
        params = initial_params

    if not isinstance(params, optax.LookaheadParams) and optimizer_uses_lookahead:
        initial_params = optax.LookaheadParams(params, params)
    else:
        initial_params = params
    del params

    opt_state = optimizer.init(initial_params)

    step_fn = _build_step_fn(
        default_loss_fn,
        network.apply,
        initial_state,
        pmap_size,
        vmap_size,
        optimizer,
        tbp=tbp,
    )

    eval_fn = _build_eval_fn(
        default_metrices, network.apply, initial_state, pmap_size, vmap_size
    )

    default_callbacks = [EvalFnCallback(eval_fn)]
    callbacks_all = default_callbacks + callbacks

    loop = TrainingLoop(
        key_generator,
        generator,
        initial_params,
        opt_state,
        step_fn,
        loggers=loggers,
        callbacks=callbacks_all,
    )

    loop.run(n_episodes)
