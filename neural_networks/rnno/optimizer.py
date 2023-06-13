from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.tree_util import tree_map
from optax._src import base, numerics


class SkipIfLargeUpdatesState(NamedTuple):
    toolarge_count: jnp.array
    inner_state: Any


def _condition_skip_large_updates(updates: base.Updates, max_norm_sq: float):
    norm_sq = jnp.sum(
        jnp.array([jnp.sum(p**2) for p in jax.tree_util.tree_leaves(updates)])
    )
    # This will also return True if `norm_sq` is NaN.
    return norm_sq < max_norm_sq


def skip_large_update(
    inner: base.GradientTransformation,
    max_norm_sq: float,
    max_consecutive_toolarge: int,
) -> base.GradientTransformation:
    "Also skips NaNs."
    inner = base.with_extra_args_support(inner)

    def init(params):
        return SkipIfLargeUpdatesState(
            toolarge_count=jnp.zeros([], jnp.int32),
            inner_state=inner.init(params),
        )

    def update(updates, state: SkipIfLargeUpdatesState, params=None, **extra_args):
        inner_state = state.inner_state
        not_toolarge = _condition_skip_large_updates(updates, max_norm_sq)
        toolarge_count = jnp.where(
            not_toolarge,
            jnp.zeros([], jnp.int32),
            numerics.safe_int32_increment(state.toolarge_count),
        )

        def do_update(_):
            return inner.update(updates, inner_state, params, **extra_args)

        def reject_update(_):
            return (tree_map(jnp.zeros_like, updates), inner_state)

        updates, new_inner_state = lax.cond(
            jnp.logical_or(not_toolarge, toolarge_count > max_consecutive_toolarge),
            do_update,
            reject_update,
            operand=None,
        )

        return updates, SkipIfLargeUpdatesState(
            toolarge_count=toolarge_count,
            inner_state=new_inner_state,
        )

    return base.GradientTransformationExtraArgs(init=init, update=update)


def replace_non_finite_updates(inner: base.GradientTransformation):
    "Replace all NaN and Inf values elementwise with zeros."

    def update(updates, inner_state, params=None, **extra_args):
        # replace NaNs and Infs
        updates = tree_map(lambda arr: jnp.where(jnp.isfinite(arr), arr, 0.0), updates)
        return inner.update(updates, inner_state, params, **extra_args)

    return base.GradientTransformationExtraArgs(init=inner.init, update=update)


def adam(
    lr=3e-3,
    steps=9000,
    alpha=1e-7,
    eps=1e-4,
    clip=0.1,
    adap_clip=0.05,
    skip_large_updates_l2_norm: Optional[float] = None,
    max_consecutive_toolarge: int = 1,
):
    # works well for rnno v2
    # clip: 0.1
    # adap clip: 0.05
    # eps: 1e-4

    schedule = optax.cosine_decay_schedule(lr, steps, alpha)
    optimizer = optax.chain(
        optax.clip(clip),
        optax.adaptive_grad_clip(adap_clip),
        optax.adam(schedule, b2=0.99, eps=eps),
    )

    optimizer = replace_non_finite_updates(optimizer)

    if skip_large_updates_l2_norm is not None:
        optimizer = skip_large_update(
            optimizer, skip_large_updates_l2_norm, max_consecutive_toolarge
        )

    optimizer = optax.lookahead(optimizer, sync_period=6, slow_step_size=0.7)
    return optimizer


def ranger(
    lr=3e-3,
    steps=1e4,
    alpha=1e-6,
    clipping=0.4,
    weight_decay=0.001,
    b1=0.9,
    b2=0.99,
    sync_period=5,
    slow_step_size=0.4,
):
    schedule = optax.cosine_decay_schedule(lr, steps, alpha)
    optimizer = optax.chain(
        optax.adaptive_grad_clip(clipping),
        optax.centralize(),
        optax.add_decayed_weights(weight_decay),
        optax.radam(learning_rate=schedule, b1=b1, b2=b2, eps=1e-6),
    )
    optimizer = optax.lookahead(
        optimizer, sync_period=sync_period, slow_step_size=slow_step_size
    )
    return optimizer
