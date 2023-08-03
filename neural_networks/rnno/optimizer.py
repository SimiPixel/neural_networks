from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp
import optax
from jax import lax
from jax.tree_util import tree_map
from optax._src import base, numerics
from optax._src.transform import AddNoiseState, add_noise


class BelowOrZero(NamedTuple):
    count: jnp.array
    inner_state: Any


def below_or_zero(
    inner: base.GradientTransformation,
    threshold: float,
    warmup: int = 0,
) -> base.GradientTransformation:
    "Sets all gradients element-wise to zero if abs(gradient) > `threshold`"
    inner = base.with_extra_args_support(inner)

    def init(params):
        return BelowOrZero(
            count=jnp.zeros([], jnp.int32),
            inner_state=inner.init(params),
        )

    def update(updates, state: BelowOrZero, params=None, **extra_args):
        inner_state = state.inner_state

        def set_to_zero(updates):
            return tree_map(
                lambda arr: jnp.where(jnp.abs(arr) > threshold, 0.0, arr), updates
            )

        updates = jax.lax.cond(
            state.count >= warmup, set_to_zero, lambda updates: updates, updates
        )

        updates, new_inner_state = inner.update(
            updates, inner_state, params, **extra_args
        )

        return updates, BelowOrZero(
            count=numerics.safe_int32_increment(state.count),
            inner_state=new_inner_state,
        )

    return base.GradientTransformationExtraArgs(init=init, update=update)


class SkipIfLargeUpdatesState(NamedTuple):
    toolarge_count: jnp.array
    count: jnp.array
    inner_state: Any
    add_noise_state: AddNoiseState


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
    warmup: int = 0,
    disturb_if_skip: bool = False,
    disturb_adaptive: bool = False,
    eta: float = 0.01,
    gamma: float = 0.55,
    seed: int = 0,
) -> base.GradientTransformation:
    "Also skips NaNs."
    inner = base.with_extra_args_support(inner)

    if disturb_adaptive:
        raise NotImplementedError

    add_noise_transform = add_noise(eta, gamma, seed)

    def init(params):
        return SkipIfLargeUpdatesState(
            toolarge_count=jnp.zeros([], jnp.int32),
            count=jnp.zeros([], jnp.int32),
            inner_state=inner.init(params),
            add_noise_state=add_noise_transform.init(params),
        )

    def update(updates, state: SkipIfLargeUpdatesState, params=None, **extra_args):
        inner_state = state.inner_state
        not_toolarge = _condition_skip_large_updates(updates, max_norm_sq)
        toolarge_count = jnp.where(
            not_toolarge,
            jnp.zeros([], jnp.int32),
            numerics.safe_int32_increment(state.toolarge_count),
        )

        def do_update(updates):
            updates, new_inner_state = inner.update(
                updates, inner_state, params, **extra_args
            )
            return updates, new_inner_state, state.add_noise_state

        def reject_update(updates):
            if disturb_if_skip:
                updates, new_add_noise_state = add_noise_transform.update(
                    updates, state.add_noise_state, params
                )
            else:
                updates, new_add_noise_state = (
                    tree_map(jnp.zeros_like, updates),
                    state.add_noise_state,
                )
            return updates, inner_state, new_add_noise_state

        updates, new_inner_state, new_add_noise_state = lax.cond(
            jnp.logical_or(
                jnp.logical_or(not_toolarge, toolarge_count > max_consecutive_toolarge),
                state.count < warmup,
            ),
            do_update,
            reject_update,
            updates,
        )

        return updates, SkipIfLargeUpdatesState(
            toolarge_count=toolarge_count,
            count=numerics.safe_int32_increment(state.count),
            inner_state=new_inner_state,
            add_noise_state=new_add_noise_state,
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
    large_updates_warmup: int = 0,
    disturb_if_skip: bool = False,
    eta: float = 0.01,
    gamma: float = 0.55,
    seed: int = 0,
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
            optimizer,
            skip_large_updates_l2_norm,
            max_consecutive_toolarge,
            large_updates_warmup,
            disturb_if_skip,
            eta=eta,
            gamma=gamma,
            seed=seed,
        )

    optimizer = optax.lookahead(optimizer, sync_period=6, slow_step_size=0.7)
    return optimizer


def adam_norm_clip(
    lr=3e-3,
    steps=9000,
    alpha=1e-7,
    eps=1e-4,
    norm_clip=0.5,
    skip_large_updates_l2_norm: Optional[float] = None,
    max_consecutive_toolarge: int = 1,
    large_updates_warmup: int = 0,
):
    # works well for rnno v2
    # clip: 0.1
    # adap clip: 0.05
    # eps: 1e-4

    schedule = optax.cosine_decay_schedule(lr, steps, alpha)
    optimizer = optax.chain(
        optax.clip_by_global_norm(norm_clip),
        optax.adam(schedule, b2=0.99, eps=eps),
    )

    optimizer = replace_non_finite_updates(optimizer)

    if skip_large_updates_l2_norm is not None:
        optimizer = skip_large_update(
            optimizer,
            skip_large_updates_l2_norm,
            max_consecutive_toolarge,
            large_updates_warmup,
        )

    optimizer = optax.lookahead(optimizer, sync_period=6, slow_step_size=0.7)
    return optimizer


def adam_below_zero(
    lr=3e-3,
    steps=9000,
    alpha=1e-7,
    eps=1e-4,
    clip=0.5,
    warmup: int = 0,
):
    schedule = optax.cosine_decay_schedule(lr, steps, alpha)
    optimizer = optax.adam(schedule, b2=0.99, eps=eps)
    optimizer = optax.lookahead(optimizer, sync_period=6, slow_step_size=0.7)
    optimizer = below_or_zero(optimizer, clip, warmup)
    optimizer = replace_non_finite_updates(optimizer)

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
