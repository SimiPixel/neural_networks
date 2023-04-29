from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import joblib
from x_xy import maths

N = 10287
T = 6000


dustin_exp_xml = r"""
<x_xy model="dustin_exp">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg1" joint="free">
            <body name="seg2" joint="ry">
                <body name="seg3" joint="rz"></body>
            </body>
        </body>
    </worldbody>
</x_xy>
"""


def dustin_exp_Xy() -> Tuple[jax.Array, jax.Array]:
    start_indices = jnp.array([start for start in range(3000, 4200, 150)])

    dd = joblib.load(Path(__file__).parent.resolve().joinpath("dustin_exp.joblib"))
    dd = jax.tree_map(jnp.asarray, dd)

    qrel = lambda q1, q2: maths.quat_mul(maths.quat_inv(q1), q2)
    X = {
        "seg1": {"acc": dd["acc1"], "gyr": dd["gyr1"]},
        "seg3": {"acc": dd["acc3"], "gyr": dd["gyr3"]},
    }
    y = {"seg2": qrel(dd["q1"], dd["q2"]), "seg3": qrel(dd["q2"], dd["q3"])}

    data = X, y

    @jax.vmap
    def extract_windows(start):
        return jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_slice_in_dim(arr, start, T), data
        )

    return extract_windows(start_indices)
