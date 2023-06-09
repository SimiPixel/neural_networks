from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import tree_utils
import x_xy
from x_xy import maths
from x_xy.utils import distribute_batchsize, expand_batchsize

from neural_networks.io_params import save
from neural_networks.logging import Logger, NeptuneLogger, flatten_dict
from neural_networks.rnno import dustin_exp_Xy
from neural_networks.rnno.training_loop import TrainingLoop, TrainingLoopCallback

from .optimizer import adam

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


def _warm_up_doesnot_count(arr):
    return arr[:, 500:]


default_metrices_dustin_exp = {
    "mae_deg": (
        lambda q, qhat: maths.angle_error(q, qhat),
        lambda arr: jnp.rad2deg(jnp.mean(_warm_up_doesnot_count(arr), axis=(0, 1))),
    ),
    "rmse_deg": (
        lambda q, qhat: maths.angle_error(q, qhat) ** 2,
        # we reduce time at axis=1, and batchsize at axis=0
        lambda arr: jnp.rad2deg(
            jnp.mean(jnp.sqrt(jnp.mean(_warm_up_doesnot_count(arr), axis=1)), axis=0)
        ),
    ),
    "q90_ae_deg": (
        lambda q, qhat: maths.angle_error(q, qhat),
        lambda arr: jnp.rad2deg(
            jnp.mean(jnp.quantile(_warm_up_doesnot_count(arr), 0.90, axis=1), axis=0)
        ),
    ),
}

default_loss_fn = lambda q, qhat: maths.angle_error(q, qhat) ** 2


def _build_eval_fn(
    eval_metrices: dict[str, Tuple[Callable, Callable]],
    apply_fn,
    initial_state,
    pmap_size,
    vmap_size,
):
    """Build function that evaluates the filter performance.
    `initial_state` has shape (pmap, vmap, state_dim)"""

    def eval_fn(params, state, X, y):
        yhat, _ = jax.vmap(apply_fn, in_axes=(None, 0, 0))(params, state, X)

        values = {}
        for metric_name, (metric_fn, reduce_fn) in eval_metrices.items():
            assert (
                metric_name not in values
            ), f"The metric identitifier {metric_name} is not unique"

            pipe = lambda q, qhat: reduce_fn(jax.vmap(jax.vmap(metric_fn))(q, qhat))
            values.update({metric_name: jax.tree_map(pipe, y, yhat)})

        return values

    @partial(jax.pmap, in_axes=(None, 0, 0, 0), out_axes=None, axis_name="devices")
    def pmapped_eval_fn(params, state, X, y):
        pmean = lambda arr: jax.lax.pmean(arr, axis_name="devices")
        values = eval_fn(params.slow, state, X, y)
        return pmean(values)

    def expand_then_pmap_eval_fn(params, X, y):
        X, y = expand_batchsize((X, y), pmap_size, vmap_size)
        return pmapped_eval_fn(params, initial_state, X, y)

    return expand_then_pmap_eval_fn


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
        (loss, state), grads = loss_fn(params.fast, state, X, y)
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


class EvalFnCallback(TrainingLoopCallback):
    def __init__(self, eval_fn):
        self.eval_fn = eval_fn

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        metrices.update(self.eval_fn(params, sample_eval[0], sample_eval[1]))


class DustinExperiment(TrainingLoopCallback):
    def __init__(
        self,
        network: hk.TransformedWithState,
        eval_dustin_exp_every: int = -1,
        metric_identifier: str = "dustin_exp",
        anchor: str = "seg1",
        q_inv: bool = True,
    ):
        self.sample = dustin_exp_Xy(anchor, q_inv)

        # build network for dustin experiment which always
        # has 3 segments; Needs its own state
        # delete batchsize dimension for init of params
        consume = jax.random.PRNGKey(1)
        _, initial_state_dustin = network.init(
            consume, tree_utils.tree_slice(self.sample[0], 0)
        )
        batchsize = tree_utils.tree_shape(self.sample)
        initial_state_dustin = _repeat_state(initial_state_dustin, batchsize)
        self.eval_fn = _build_eval_fn(
            default_metrices_dustin_exp,
            network.apply,
            initial_state_dustin,
            *distribute_batchsize(batchsize),
        )
        self.eval_dustin_exp_every = eval_dustin_exp_every
        self.metric_identifier = metric_identifier

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        if self.eval_dustin_exp_every == -1:
            return

        if (i_episode % self.eval_dustin_exp_every) == 0:
            self.last_metrices = flatten_dict(
                {
                    self.metric_identifier: self.eval_fn(
                        params, self.sample[0], self.sample[1]
                    )
                }
            )

        metrices.update(self.last_metrices)


class SaveParamsTrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self, n_episodes: int, path_to_file: str, upload_to_neptune: bool = True
    ):
        self.n_episodes = n_episodes
        self.path_to_file = str(
            Path(path_to_file).expanduser().with_suffix("").with_suffix(".pickle")
        )
        self._upload_to_neptune = upload_to_neptune

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        if i_episode != self.n_episodes - 1:
            return

        # params is Lookahead object
        save(params.slow, self.path_to_file, overwrite=True)

        if self._upload_to_neptune:
            for logger in loggers:
                if isinstance(logger, NeptuneLogger):
                    logger.log_params(self.path_to_file)
                    break
            else:
                raise Exception(f"No `NeptuneLogger` was found in {loggers}")


class LogGradsTrainingLoopCallBack(TrainingLoopCallback):
    def __init__(self, print=False) -> None:
        self.print = print

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        gradient_log = {}
        for i, grads_tbp in enumerate(grads):
            grads_flat = tree_utils.batch_concat(grads_tbp, num_batch_dims=0)
            grads_max = jnp.max(jnp.abs(grads_flat))
            grads_norm = jnp.linalg.norm(grads_flat)
            gradient_log[f"grads_tbp_{i}_max"] = grads_max
            gradient_log[f"grads_tbp_{i}_l2norm"] = grads_norm

        if print:
            print(gradient_log)

        metrices.update(gradient_log)


def _repeat_state(state, repeats: int):
    pmap_size, vmap_size = distribute_batchsize(repeats)
    return jax.vmap(jax.vmap(lambda _: state))(jnp.zeros((pmap_size, vmap_size)))


key_generator, key_network = jax.random.split(jax.random.PRNGKey(0))


def train(
    generator: Callable,
    n_episodes: int,
    network: hk.TransformedWithState,
    optimizer=adam(),
    tbp: int = 1000,
    network_dustin=None,
    loggers: list[Logger] = [],
    callbacks: list[TrainingLoopCallback] = [],
    initial_params: Optional[dict] = None,
    add_dustin_exp_callback: bool = True,
    key_network: jax.random.PRNGKey = key_network,
    key_generator: jax.random.PRNGKey = key_generator,
):
    """Trains RNNO

    Args:
        generator (Callable): output of the rcmg-module
        n_episodes (int): number of episodes to train for
        network (hk.TransformedWithState): RNNO network
        optimizer (_type_, optional): optimizer, see optimizer.py module
        tbp (int, optional): Truncated backpropagation through time step size
        network_dustin (_type_, optional): RNNO network used for evaluation on dustin's
            exp. Only RNNOv2 has the ability to be trained on a four segment chain,
            yet be evaluated on a three segment setup.
        loggers: list of Loggers used to log the training progress.
        callbacks: callbacks of the TrainingLoop.
        initial_params: If given uses as initial parameters.
        add_dustin_exp_callback: If `True` appends a callback that evalutaes the
            Dustin experiment.
        key_network: PRNG Key that inits the network state and parameters.
        key_generator: PRNG Key that inits the data stream of the generator.
    """

    if network_dustin is None:
        network_dustin = network

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

    if not isinstance(params, optax.LookaheadParams):
        initial_params = optax.LookaheadParams(params, params)

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

    if add_dustin_exp_callback:
        default_callbacks = [
            EvalFnCallback(eval_fn),
            DustinExperiment(network_dustin, 5),
        ]
    else:
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
