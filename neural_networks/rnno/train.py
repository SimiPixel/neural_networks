from dataclasses import dataclass, field
from functools import partial
from types import SimpleNamespace
from typing import Callable

import jax
import jax.numpy as jnp
import optax
import tree_utils
from x_xy import maths
from x_xy.rcmg import distribute_batchsize, expand_batchsize

from neural_networks.logging import Logger, flatten_dict, n_params
from neural_networks.rnno import generator_dustin_exp
from neural_networks.rnno.optimizer import adam

default_metrices = {
    "rmse_deg": (
        lambda q, qhat: angle_error(q, qhat) ** 2,
        # we reduce time at axis=1, and batchsize at axis=0
        lambda arr: jnp.rad2deg(jnp.mean(jnp.sqrt(jnp.mean(arr, axis=1)), axis=0)),
    ),
    "mae_deg": (
        lambda q, qhat: jnp.abs(angle_error(q, qhat)),
        lambda arr: jnp.rad2deg(jnp.mean(arr, axis=(0, 1))),
    ),
    "q90_ae_deg": (
        lambda q, qhat: jnp.abs(angle_error(q, qhat)),
        lambda arr: jnp.rad2deg(jnp.mean(jnp.quantile(arr, 0.90, axis=1), axis=0)),
    ),
    "q99_ae_deg": (
        lambda q, qhat: jnp.abs(angle_error(q, qhat)),
        lambda arr: jnp.rad2deg(jnp.mean(jnp.quantile(arr, 0.99, axis=1), axis=0)),
    ),
}


@dataclass
class RNNO_Config:
    n_episodes: int
    tbp: int = 1000
    key: jax.Array = jax.random.PRNGKey(1)
    loss_metric: Callable = lambda q, qhat: angle_error(q, qhat) ** 2
    eval_metrices: dict[str, tuple] = field(default_factory=lambda: default_metrices)


def train(
    generator: Callable,
    network: Callable,
    config: RNNO_Config,
    loggers: list[Logger],
    eval_dustin_exp_every: int = -1,
    network_dustin=None,
    callbacks: list[Callable] = [],
    jit: bool = True,
) -> Callable:

    if jit:
        generator = jax.jit(generator)

    key, consume = jax.random.split(config.key)
    # initialize params, state, opt_state
    sample_toy = generator(config.key)
    batchsize = tree_utils.tree_shape(sample_toy, 0)
    N = tree_utils.tree_shape(sample_toy, 1)

    # delete batchsize dimension for init of params
    params, state = network.init(
        consume, jax.tree_map(lambda arr: arr[0], sample_toy["X"])
    )

    for logger in loggers:
        logger.log(dict(n_params=n_params(params), batchsize=batchsize))

    # we assume every optimizer uses lookahead
    params = optax.LookaheadParams(params, params)
    opt = _build_optimizer(config.n_episodes, config.tbp, N)
    opt_state = opt.init(params)

    pmap_size, vmap_size = distribute_batchsize(batchsize)

    # build step fn
    step_fn = _build_step_fn(
        config.loss_metric,
        network.apply,
        state,
        pmap_size,
        vmap_size,
        opt,
        config.tbp,
        N,
    )

    # build eval fn
    eval_fn = _build_eval_fn(
        config.eval_metrices, network.apply, state, pmap_size, vmap_size
    )

    # build eval fn for dustin exp
    if eval_dustin_exp_every != -1:
        assert (
            network_dustin is not None
        ), "Can only evaluate dustin experiment if network is given"
        sample_dustin_exp = generator_dustin_exp()
        # build network for dustin experiment which always
        # has 3 segments; Needs its own state
        # delete batchsize dimension for init of params
        key, consume = jax.random.split(key)
        _, state_network_dustin = network_dustin.init(
            consume, jax.tree_map(lambda arr: arr[0], sample_dustin_exp["X"])
        )
        pmap_size_dustin, vmap_size_dustin = distribute_batchsize(
            tree_utils.tree_shape(sample_dustin_exp)
        )
        eval_fn_dustin_exp = _build_eval_fn(
            config.eval_metrices,
            network_dustin.apply,
            state_network_dustin,
            pmap_size_dustin,
            vmap_size_dustin,
        )

    # compile step fn and eval fn
    if jit:
        step_fn = jax.jit(step_fn)
        eval_fn = jax.jit(eval_fn)
        if eval_dustin_exp_every != -1:
            eval_fn_dustin_exp = jax.jit(eval_fn_dustin_exp)

    # start training loop
    key, consume = jax.random.split(key)
    sample_eval = generator(consume)

    for i_episode in range(config.n_episodes):
        sample_train = sample_eval
        key, consume = jax.random.split(key)
        sample_eval = generator(consume)

        params, opt_state, loss = step_fn(
            params, opt_state, sample_train["X"], sample_train["y"]
        )
        metrices = eval_fn(params, sample_eval["X"], sample_eval["y"])

        metrices.update(loss)

        if eval_dustin_exp_every != -1:
            if (i_episode % eval_dustin_exp_every) == 0:
                dustin_exp_eval_metrices = flatten_dict(
                    {
                        "dustin_exp": eval_fn_dustin_exp(
                            params, sample_dustin_exp["X"], sample_dustin_exp["y"]
                        )
                    }
                )
            metrices.update(dustin_exp_eval_metrices)

        for callback in callbacks:
            # TODO
            callback(i_episode, metrices, params.slow, network.apply)

        for logger in loggers:
            logger.log(metrices)

            if i_episode == (config.n_episodes - 1):
                logger.close()

    @jax.jit
    def final_predict(X):
        """No batchsize!"""
        return network.apply(params.slow, state, X)

    @jax.jit
    def final_eval(X, y):
        """No batchsize!"""
        X, y = tree_utils.add_batch_dim((X, y))
        return _build_eval_fn(config.eval_metrices, network.apply, state, 1, 1)(
            params.slow, X, y
        )

    return params.slow, SimpleNamespace(predict=final_predict, eval=final_eval)


def _build_optimizer(n_episodes, tbp, N):
    steps = int(N / tbp)
    return adam(steps=n_episodes * steps)


def _build_vmap_vmap_metric_fn(metric_fn: Callable, reduce_fn: Callable):
    """Builds a metric function that is mapped accross batchsize on device and time.
    - metric_fn: (y, yhat) -> point_estimate
    - reduce_fn: (batchsize / n_devices, n_timesteps, point_estimate) -> point_estimate
    """

    def vmap_vmap_metric_fn(y, yhat):
        metric_per_node = jax.tree_map(jax.vmap(jax.vmap(metric_fn)), y, yhat)
        point_estimate_per_node = jax.tree_map(reduce_fn, metric_per_node)
        return point_estimate_per_node

    return vmap_vmap_metric_fn


def _build_eval_fn(eval_metrices, apply_fn, state, pmap_size, vmap_size):
    def eval_fn(params, X, y):
        X, y = expand_batchsize((X, y), pmap_size, vmap_size)

        metrices_values = {}
        for metric_name, (metric_fn, reduce_fn) in eval_metrices.items():

            @jax.pmap
            def point_estimate_per_node(X, y):
                yhat = jax.vmap(lambda X: apply_fn(params.slow, state, X)[0])(X)
                metric_per_node = _build_vmap_vmap_metric_fn(metric_fn, reduce_fn)(
                    y, yhat
                )
                return metric_per_node

            point_estimate = jax.tree_util.tree_map(
                lambda arr: jnp.mean(arr, axis=0), point_estimate_per_node(X, y)
            )
            metrices_values.update({metric_name: point_estimate})

        return metrices_values

    return eval_fn


def _build_step_fn(
    metric_fn, apply_fn, initial_state, pmap_size, vmap_size, optimizer, tbp, N
):

    # repeat state along batchsize
    initial_state = jax.vmap(jax.vmap(lambda _: initial_state))(
        jnp.zeros((pmap_size, vmap_size))
    )
    reduce_fn = lambda arr: jnp.mean(arr, axis=(0, 1))

    @partial(jax.pmap, in_axes=(None, 0, 0, 0))
    def loss_fn(params, state, X, y):
        yhat, state = jax.vmap(lambda state, X: apply_fn(params, state, X))(state, X)
        loss_per_node = _build_vmap_vmap_metric_fn(metric_fn, reduce_fn)(y, yhat)
        mean_loss = jnp.mean(tree_utils.batch_concat(loss_per_node, 0), axis=0)
        return mean_loss, state

    @partial(jax.value_and_grad, has_aux=True)
    def grad_loss_fn(params, state, X, y):
        loss_with_pmap_dim, state = loss_fn(params, state, X, y)
        return jnp.mean(loss_with_pmap_dim, axis=0), state

    def step_fn(params, opt_state, X, y):
        X, y = expand_batchsize((X, y), pmap_size, vmap_size)

        nonlocal initial_state

        state = initial_state
        for X_tbp, y_tbp in tree_utils.tree_split((X, y), int(N / tbp), axis=-2):
            (loss_value, state), grads = grad_loss_fn(params.fast, state, X_tbp, y_tbp)
            state = jax.lax.stop_gradient(state)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

        return params, opt_state, {"loss": loss_value}

    return step_fn


def angle_error(q, qhat):
    return maths.quat_angle(maths.quat_mul(maths.quat_inv(q), qhat))
