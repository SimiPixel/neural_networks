from collections import deque
from functools import partial
import os
from pathlib import Path
import time
from typing import Callable, Optional, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
from optax import LookaheadParams
import tree_utils
from x_xy import base
from x_xy import maths
from x_xy.io import load_sys_from_str
from x_xy.subpkgs import pipeline
from x_xy.utils import distribute_batchsize
from x_xy.utils import expand_batchsize
from x_xy.utils import merge_batchsize
from x_xy.utils import parse_path

from neural_networks.io_params import save
from neural_networks.logging import Logger
from neural_networks.logging import MultimediaLogger
from neural_networks.rnno import dustin_exp_xml_seg1
from neural_networks.rnno import dustin_exp_xml_seg2
from neural_networks.rnno import dustin_exp_xml_seg3
from neural_networks.rnno import dustin_exp_Xy
from neural_networks.rnno.training_loop import send_kill_run_signal
from neural_networks.rnno.training_loop import TrainingLoopCallback


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
        yhat, _ = apply_fn(params, state, X)

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
        if isinstance(params, LookaheadParams):
            slow_params = params.slow
        else:
            slow_params = params
        values = eval_fn(slow_params, state, X, y)
        return pmean(values)

    def expand_then_pmap_eval_fn(params, X, y):
        X, y = expand_batchsize((X, y), pmap_size, vmap_size)
        return pmapped_eval_fn(params, initial_state, X, y)

    return expand_then_pmap_eval_fn


class DefaultEvalFnCallback(TrainingLoopCallback):
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


def rename_keys(d: dict, rename: dict = {}):
    new_d = {}
    for key, value in d.items():
        if key in rename:
            new_d[rename[key]] = value
        else:
            new_d[key] = d[value]
    return new_d


class DustinExperiment(TrainingLoopCallback):
    def __init__(
        self,
        rnno_fn: hk.TransformedWithState,
        eval_dustin_exp_every: int = -1,
        metric_identifier: str = "dustin_exp",
        anchor: str = "seg1",
        q_inv: bool = True,
        with_seg2: bool = False,
    ):
        xml_str_dustin = {
            "seg1": dustin_exp_xml_seg1,
            "seg2": dustin_exp_xml_seg2,
            "seg3": dustin_exp_xml_seg3,
        }[anchor]
        network = rnno_fn(load_sys_from_str(xml_str_dustin))

        X, y = dustin_exp_Xy(anchor, q_inv, with_seg2=with_seg2)
        self.sample = X, y

        # build network for dustin experiment which always
        # has 3 segments; Needs its own state
        # delete batchsize dimension for init of params
        consume = jax.random.PRNGKey(1)
        _, initial_state_dustin = network.init(consume, self.sample[0])
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


def default_metrices_eval_xy(warmup: int, cooloff: int):
    return {
        "mae_deg": (
            lambda q, qhat: maths.angle_error(q, qhat),
            lambda arr: jnp.rad2deg(jnp.mean(arr[:, warmup:-cooloff], axis=(0, 1))),
        ),
        "q90_ae_deg": (
            lambda q, qhat: maths.angle_error(q, qhat),
            lambda arr: jnp.rad2deg(
                jnp.mean(jnp.quantile(arr[:, warmup:-cooloff], 0.90, axis=1), axis=0)
            ),
        ),
    }


class EvalXyTrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        network: hk.TransformedWithState,
        X: dict,
        y: dict,
        metric_identifier: str,
        eval_every: int = 5,
        warmup: int = 500,
        cooloff: int = 1,
    ):
        "X, y is batched."
        assert cooloff > 0

        self.sample = (X, y)

        # delete batchsize dimension for init of state
        consume = jax.random.PRNGKey(1)
        _, state = network.init(consume, tree_utils.tree_slice(self.sample[0], 0))
        batchsize = tree_utils.tree_shape(self.sample)
        state = _repeat_state(state, batchsize)
        self.eval_fn = _build_eval_fn(
            default_metrices_eval_xy(warmup, cooloff),
            network.apply,
            state,
            *distribute_batchsize(batchsize),
        )
        self.eval_every = eval_every
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
        if self.eval_every == -1:
            return

        if (i_episode % self.eval_every) == 0:
            self.last_metrices = {
                self.metric_identifier: self.eval_fn(
                    params, self.sample[0], self.sample[1]
                )
            }

        metrices.update(self.last_metrices)


def _build_eval_fn2(
    eval_metrices: dict[str, Tuple[Callable, Callable, Callable]],
    apply_fn,
    initial_state,
    pmap_size,
    vmap_size,
):
    @partial(jax.pmap, in_axes=(None, 0, 0))
    def pmap_vmap_apply(params, initial_state, X):
        return apply_fn(params, initial_state, X)[0]

    def eval_fn(params, X, y):
        params = params.slow if isinstance(params, LookaheadParams) else params
        X = expand_batchsize(X, pmap_size, vmap_size)
        yhat = pmap_vmap_apply(params, initial_state, X)
        yhat = merge_batchsize(yhat, pmap_size, vmap_size)

        values, post_reduce1 = {}, {}
        for metric_name, (metric_fn, reduce_fn1, reduce_fn2) in eval_metrices.items():
            assert (
                metric_name not in values
            ), f"The metric identitifier {metric_name} is not unique"

            reduce1_errors_fn = lambda q, qhat: reduce_fn1(
                jax.vmap(jax.vmap(metric_fn))(q, qhat)
            )
            post_reduce1_errors = jax.tree_map(reduce1_errors_fn, y, yhat)
            values.update({metric_name: jax.tree_map(reduce_fn2, post_reduce1_errors)})
            post_reduce1.update({metric_name: post_reduce1_errors})

        return values, post_reduce1

    return eval_fn


class EvalXy2TrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        exp_name: str,
        rnno_fn,
        sys_noimu,
        eval_metrices: dict[str, Tuple[Callable, Callable, Callable]],
        X: dict,
        y: dict,
        xs: base.Transform,
        sys_xs,
        metric_identifier: str,
        render_plot_metric: str,
        eval_every: int = 5,
        render_plot_every: int = 50,
        maximal_error: bool | list[bool] = True,
        plot: bool = False,
        render: bool = False,
        upload: bool = True,
        save2disk: bool = False,
        render_0th_epoch: bool = True,
        verbose: bool = True,
        show_cs: bool = False,
        show_cs_root: bool = True,
    ):
        "X, y is batched."

        network = rnno_fn(sys_noimu)
        self.sys_noimu, self.sys_xs = sys_noimu, sys_xs
        self.X, self.y, self.xs = X, y, xs
        self.plot, self.render = plot, render
        self.upload = upload
        self.save2disk = save2disk
        self.render_plot_metric = render_plot_metric
        self.maximal_error = (
            maximal_error if isinstance(maximal_error, list) else [maximal_error]
        )
        self.rnno_fn = rnno_fn
        self.path = f"~/experiments/{exp_name}"

        # delete batchsize dimension for init of state
        consume = jax.random.PRNGKey(1)
        _, initial_state = network.init(consume, X)
        batchsize = tree_utils.tree_shape(X)
        self.eval_fn = _build_eval_fn2(
            eval_metrices,
            network.apply,
            _repeat_state(initial_state, batchsize),
            *distribute_batchsize(batchsize),
        )
        self.eval_every = eval_every
        self.render_plot_every = render_plot_every
        self.metric_identifier = metric_identifier
        self.render_0th_epoch = render_0th_epoch
        self.verbose = verbose
        self.show_cs, self.show_cs_root = show_cs, show_cs_root

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ):
        self._params = params
        self._loggers = loggers
        self.i_episode = i_episode

        if self.eval_every == -1:
            return

        if (i_episode % self.eval_every) == 0:
            point_estimates, self.per_seq = self.eval_fn(params, self.X, self.y)
            self.last_metrices = {self.metric_identifier: point_estimates}
        metrices.update(self.last_metrices)

        if (i_episode % self.render_plot_every) == 0:
            if i_episode != 0 or self.render_0th_epoch:
                self._render_plot()

    def close(self):
        self._render_plot()

    def _render_plot(self):
        if not self.plot and not self.render:
            return

        if isinstance(self._params, LookaheadParams):
            params = self._params.slow
        else:
            params = self._params

        for maximal_error in self.maximal_error:
            reduce = jnp.argmax if maximal_error else jnp.argmin
            idx = reduce(
                jnp.mean(
                    tree_utils.batch_concat(self.per_seq[self.render_plot_metric]),
                    axis=-1,
                )
            )
            X, y, xs = tree_utils.tree_slice((self.X, self.y, self.xs), idx)

            def filename(prefix: str):
                return (
                    f"{prefix}_{self.metric_identifier}_{self.render_plot_metric}_"
                    f"idx={idx}_episode={self.i_episode}_maxError={int(maximal_error)}"
                )

            render_path = parse_path(
                self.path,
                "videos",
                filename("animation"),
                extension="mp4",
            )

            if self.verbose:
                print(f"--- EvalFnCallback {self.metric_identifier} --- ")

            pipeline.predict(
                self.sys_noimu,
                self.rnno_fn,
                X,
                y,
                xs,
                self.sys_xs,
                params,
                plot=self.plot,
                render=self.render,
                render_path=render_path,
                verbose=self.verbose,
                show_cs=self.show_cs,
                show_cs_root=self.show_cs_root,
            )

            plot_path = parse_path(
                self.path,
                "plots",
                filename("plot"),
                extension="png",
            )
            if self.plot:
                import matplotlib.pyplot as plt

                plt.savefig(plot_path, dpi=300)
                plt.close()

            if self.upload:
                logger = _find_multimedia_logger(self._loggers)
                if self.render:
                    logger.log_video(render_path, step=self.i_episode)
                if self.plot:
                    logger.log_image(plot_path)

            if not self.save2disk:
                for path in [render_path, plot_path]:
                    if Path(path).exists():
                        os.system(f"rm {path}")


class SaveParamsTrainingLoopCallback(TrainingLoopCallback):
    def __init__(
        self,
        path_to_file: str,
        upload: bool = True,
        last_n_params: int = 1,
        slow_and_fast: bool = False,
    ):
        self.path_to_file = parse_path(path_to_file, extension="pickle")
        self.upload = upload
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
        if not self.slow_and_fast and isinstance(params, LookaheadParams):
            params = params.slow

        self._params.append(params)
        self._loggers = loggers

    def close(self):
        params = list(self._params)
        if len(params) == 1:
            params = params[0]

        save(params, self.path_to_file, overwrite=True)

        if self.upload:
            logger = _find_multimedia_logger(self._loggers)
            logger.log_params(self.path_to_file)


def _find_multimedia_logger(loggers):
    for logger in loggers:
        if isinstance(logger, MultimediaLogger):
            return logger
    raise Exception(f"Neither `NeptuneLogger` nor `WandbLogger` was found in {loggers}")


class LogGradsTrainingLoopCallBack(TrainingLoopCallback):
    def __init__(
        self,
        print=False,
        kill_if_larger: Optional[float] = None,
        consecutive_larger: int = 1,
    ) -> None:
        self.print = print
        self.kill_if_larger = kill_if_larger
        self.consecutive_larger = consecutive_larger
        self.last_larger = deque(maxlen=consecutive_larger)

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
            if self.kill_if_larger is not None:
                if grads_norm > self.kill_if_larger:
                    self.last_larger.append(True)
                else:
                    self.last_larger.append(False)
                if all(self.last_larger):
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
        if isinstance(params, LookaheadParams):
            fast_params = params.fast
        else:
            fast_params = params
        params_fast_flat = tree_utils.batch_concat(fast_params, num_batch_dims=0)
        params_is_nan = jnp.any(jnp.isnan(params_fast_flat))

        if params_is_nan:
            send_kill_run_signal()

        if params_is_nan and self.print:
            print(
                f"Parameters have converged to NaN at step {i_episode}. Exiting run.."
            )


class LogEpisodeTrainingLoopCallback(TrainingLoopCallback):
    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        metrices.update({"i_episode": i_episode})


class TimingKillRunCallback(TrainingLoopCallback):
    def __init__(self, max_run_time_seconds: float) -> None:
        self.max_run_time_seconds = max_run_time_seconds
        self.t0 = time.time()

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: LookaheadParams,
        grads: list[dict],
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        runtime = time.time() - self.t0
        if runtime > self.max_run_time_seconds:
            runtime_h = runtime / 3600
            print(f"Run is killed due to timing. Current runtime is {runtime_h}h.")
            send_kill_run_signal()


def make_utility_callbacks(
    params_path=None,
    kill_if_larger: Optional[float] = None,
    consecutive_larger: int = 1,
) -> list[TrainingLoopCallback]:
    callbacks = [
        LogGradsTrainingLoopCallBack(
            kill_if_larger=kill_if_larger, consecutive_larger=consecutive_larger
        ),
        NanKillRunCallback(),
        TimingKillRunCallback(23.0 * 3600),
        LogEpisodeTrainingLoopCallback(),
    ]
    if params_path is not None:
        callbacks.append(SaveParamsTrainingLoopCallback(params_path))
    return callbacks


def _repeat_state(state, repeats: int):
    pmap_size, vmap_size = distribute_batchsize(repeats)
    return jax.vmap(jax.vmap(lambda _: state))(jnp.zeros((pmap_size, vmap_size)))
