# RNN-based Observer (RNNO)

**Currently only supports kinematic chains. No branching!**

## Installation

Create a new conda-env with `Python=3.10`.
Then,
- `pip install jaxlib==0.4.3 jax==0.4.4` 
- `pip install git+https://github.com/SimiPixel/neural_networks.git`

---

This repository hosts RNNO in two versions.

RNNOv1 -> `network.py`

RNNOv2 -> `network_local.py`

We will work with the v2 version mostly (because i believe it is better).

It also hosts experimental data to validate RNNO against (also during training process of RNNO).

## What the RNNO_v1/2 networks expect as `X` and returns as `y`
Both `rnno_network` and `rnno_network_local` expect the input `X` to be

```python
# X is a dict of the time-series of measurement of both outer IMUs
# So suppose you have a three segment kinematic chain, that means N=3
# segment0, segment1, segment2
# segment0 has IMU0
# segment1 has no measurements; Note that it is missing below
# segment2 (N-1) has IMU1
X = {
    0: {
        "acc": jax.Array,
        "gyr": jax.Array,
    },
    N-1: {
        "acc": jax.Array,
        "gyr": jax.Array,
    }
    # the jax.Arrays should be of shape (n_timesteps, 3)
    # where the measurements match accross nodes (= segments)
    # so currently mixing 9D and 6D IMUs would not be possible
    # Here, this would be for 6D IMUs
}
```

and what it returns is
```python
# y is a dict of the time-series of the relative quaternions from the
# segment to its parent.
# So suppose we have a three segment kinematic chain
# for segment1 the parent is segment0, and for segment2 the parent
# is segment1. Note that segment0 has no parent (we only estimate the *relative* pose)
# it's also missing below
y = {
    # the jax.Array should be of shape (n_timesteps, 4)
    1: jax.Array
    2: jax.Array
    ...
    N-1: jax.Array 
}
```
## Both RNNO_v1/2 networks are maps from

```python
network = rnno_network()
# initialize the network parameters and the initial state 
# using a random seed `key`
params, state = network.init(key, X)
# then we can call the network with 
y = network.apply(params, state, X)
```
where `X` has no batchsize dimension. Batching is done via `jax.vmap`

## So where does `X,y` come from?
Since we train RNNO on *random* chain motion, there must be some generating function that, given a seed, produces some random chain motion. Records the IMU measurements and the relative pose and returns the data.

This is handled by the `x_xy.rcmg` module. Consider e.g. the snippet

```python
from x_xy.rcmg import rcmg_3seg
from jax import random

# this produces a generating function that generates random motion of a three-segment chain
batchsize = 32
# check out the source code of this function; it's quite intuitive
generator = rcmg_3Seg(
    batchsize,
    randomized_interpolation=False,
    randomized_anchors=True,
    range_of_motion=True,
    range_of_motion_method="uniform",
    Ts=0.01,  # seconds
    T=60,  # seconds
    t_min=0.15,  # min time between two generated angles
    t_max=0.75,  # max time ...
    dang_min=jnp.deg2rad(0),  # minimum angular velocity in deg/s
    dang_max=jnp.deg2rad(120),  # maximum angular velocity in deg/s
    dang_min_global=jnp.deg2rad(0),
    dang_max_global=jnp.deg2rad(60),
    dpos_min=0.001,  # speed of translation
    dpos_max=0.1,
    pos_min=-2.5,
    pos_max=+2.5,
    param_ident=None,
)

seed = 1
data = generator(random.PRNGKey(seed))

X, y = data["X"], data["y"]

# where `X` and `y` have a leading batchsize of 32
```

## Okay we have data. How do we train?

```python
from neural_networks.rnno import train, rnno_network, rnno_network_local

# rnno_network would work too
# but let's go with RNNO_v2
network = rnno_network_local(length_of_chain=3)

from x_xy.rcmg import rcmg_3seg
generator = rcmg_3Seg(batchsize=32)

# start training
n_episodes = 1500
train(generator, n_episodes, network, loggers=[])
```

## Logging the training progress
I use neptune to log runs. For this purpose make sure that the environment variable `NEPTUNE_TOKEN` is set with the token of your neptune account.
Then provide the logger like so
```python
from neural_networks.logging import NeptuneLogger
train(generator, n_epsiodes, network, loggers=[NeptuneLogger()])
```

### Bonus: Training on 4Seg, Evaluating on 3Seg (dustin's experiment)

```python
from x_xy.rcmg.rcmg_old_4Seg import rcmg_4Seg

from neural_networks.rnno.network_local import rnno_network_local
from neural_networks.rnno.train import train

batchsize = 1024

generator = rcmg_4Seg(
    batchsize,
    t_min=0.1,
    t_max=0.5,
    dang_min=0.1,
    dang_max=2.4,
)

network = rnno_network_local(length_of_chain=4)
network_dustin = rnno_network_local(length_of_chain=3)

train(
    generator,
    1500,
    network,
    network_dustin,
)
```