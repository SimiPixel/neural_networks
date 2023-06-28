from collections import deque
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils
from optax import LookaheadParams
from x_xy import maths
from x_xy.utils import distribute_batchsize, expand_batchsize

from neural_networks.io_params import save
from neural_networks.logging import Logger, NeptuneLogger
from neural_networks.rnno import dustin_exp_Xy
from neural_networks.rnno.training_loop import (
    TrainingLoopCallback,
    send_kill_run_signal,
)


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


class EvalFnCallback(TrainingLoopCallback):
    def __init__(self, eval_fn):
        self.eval_fn = eval_fn

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        metrices.update(self.eval_fn(params, sample_eval[0], sample_eval[1]))


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
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        if self.eval_dustin_exp_every == -1:
            return

        if (i_episode % self.eval_dustin_exp_every) == 0:
            self.last_metrices = {
                self.metric_identifier: self.eval_fn(
                    params, self.sample[0], self.sample[1]
                )
            }

        metrices.update(self.last_metrices)


class SaveParamsTrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        path_to_file: str,
        upload_to_neptune: bool = True,
        last_n_params: int = 1,
        slow_and_fast: bool = False,
    ):
        self.path_to_file = str(
            Path(path_to_file).expanduser().with_suffix("").with_suffix(".pickle")
        )
        self._upload_to_neptune = upload_to_neptune
        self._params = deque(maxlen=last_n_params)
        self.slow_and_fast = slow_and_fast
        self._loggers = []

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        if not self.slow_and_fast:
            params = params.slow

        self._params.append(params)
        self._loggers = loggers

    def close(self):
        # params is Lookahead object
        params = list(self._params)
        if len(params) == 1:
            params = params[0]

        save(params, self.path_to_file, overwrite=True)

        if self._upload_to_neptune:
            for logger in self._loggers:
                if isinstance(logger, NeptuneLogger):
                    logger.log_params(self.path_to_file)
                    break
            else:
                raise Exception(f"No `NeptuneLogger` was found in {self._loggers}")


class LogGradsTrainingLoopCallBack(TrainingLoopCallback):
    def __init__(self, print=False, kill_if_larger: Optional[float] = None) -> None:
        self.print = print
        self.kill_if_larger = kill_if_larger

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        gradient_log = {}
        for i, grads_tbp in enumerate(grads):
            grads_flat = tree_utils.batch_concat(grads_tbp, num_batch_dims=0)
            grads_max = jnp.max(jnp.abs(grads_flat))
            grads_norm = jnp.linalg.norm(grads_flat)
            if self.kill_if_larger is not None and grads_norm > self.kill_if_larger:
                send_kill_run_signal()
            gradient_log[f"grads_tbp_{i}_max"] = grads_max
            gradient_log[f"grads_tbp_{i}_l2norm"] = grads_norm

        if self.print:
            print(gradient_log)

        metrices.update(gradient_log)


class NanKillRunCallback(TrainingLoopCallback):
    def __init__(
        self,
        print: bool = True,
    ) -> None:
        self.print = print

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        params_fast_flat = tree_utils.batch_concat(params.fast, num_batch_dims=0)
        params_is_nan = jnp.any(jnp.isnan(params_fast_flat))

        if params_is_nan:
            send_kill_run_signal()

        if params_is_nan and self.print:
            print(
                f"Parameters have converged to NaN at step {i_episode}. Exiting run.."
            )


def _repeat_state(state, repeats: int):
    pmap_size, vmap_size = distribute_batchsize(repeats)
    return jax.vmap(jax.vmap(lambda _: state))(jnp.zeros((pmap_size, vmap_size)))