import optax


# Currently adam is hardcoded into `train.py`
def adam(lr=3e-3, steps=9000, alpha=1e-7):
    schedule = optax.cosine_decay_schedule(lr, steps, alpha)
    optimizer = optax.chain(
        optax.clip(0.2), optax.adaptive_grad_clip(0.15), optax.adam(schedule, b2=0.99)
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
