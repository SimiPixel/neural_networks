from pathlib import Path

import jax
import jax.numpy as jnp
import joblib
from x_xy import maths

N = 10287
T = 6000


def generator_dustin_exp(*args):
    start_indices = jnp.array([start + 10 for start in range(3000, 4200, 150)])

    dd = joblib.load(Path(__file__).parent.resolve().joinpath("dustin_exp.joblib"))
    dd = jax.tree_map(jnp.asarray, dd)

    # transform raw data to required format which is
    # {"X": {0: {"acc": ..., "gyr"}, 2: {...}}, "y": {2: ..., 1: ...}}

    qrel = lambda q1, q2: maths.quat_mul(maths.quat_inv(q1), q2)
    data = {
        "X": {
            0: {"acc": dd["acc1"], "gyr": dd["gyr1"]},
            2: {"acc": dd["acc3"], "gyr": dd["gyr3"]},
        },
        "y": {2: qrel(dd["q2"], dd["q3"]), 1: qrel(dd["q1"], dd["q2"])},
    }

    @jax.vmap
    def extract_windows(start):
        return jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_slice_in_dim(arr, start, T), data
        )

    return extract_windows(start_indices)
