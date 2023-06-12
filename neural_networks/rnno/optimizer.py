from functools import partial
from typing import Optional

import optax


def adam(
    lr=3e-3,
    steps=9000,
    alpha=1e-7,
    eps=1e-4,
    clip=0.1,
    adap_clip=0.05,
    skip_large_updates_l2_norm: Optional[float] = None,
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

    if skip_large_updates_l2_norm is not None:
        optimizer = optax.MultiSteps(
            optimizer,
            every_k_schedule=1,
            should_skip_update_fn=partial(
                optax.skip_large_updates, max_squared_norm=skip_large_updates_l2_norm
            ),
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
