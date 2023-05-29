from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import joblib
from x_xy import maths

N = 10287
T = 6000


dustin_exp_xml_seg1 = r"""
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

dustin_exp_xml_seg2 = r"""
<x_xy model="dustin_exp">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg2" joint="free">
            <body name="seg1" joint="ry"></body>
            <body name="seg3" joint="rz"></body>
        </body>
    </worldbody>
</x_xy>
"""

dustin_exp_xml_seg3 = r"""
<x_xy model="dustin_exp">
    <options gravity="0 0 9.81" dt="0.01"/>
    <worldbody>
        <body name="seg3" joint="free">
            <body name="seg2" joint="rz">
                <body name="seg1" joint="ry"></body>
            </body>
        </body>
    </worldbody>
</x_xy>
"""

# match default value from before
dustin_exp_xml = dustin_exp_xml_seg1


def _remove_nan_values(dd):
    # so apparently all OMC quaternion measurements have a single
    # nan value; at one single timestep; simply overwrite that one
    dd["q1"][4] = dd["q1"][3]
    dd["q2"][4] = dd["q2"][3]
    dd["q3"][5] = dd["q3"][4]
    return dd


def dustin_exp_Xy(
    anchor: str = "seg1", q_inv: bool = True
) -> Tuple[jax.Array, jax.Array]:
    start_indices = jnp.array([start for start in range(3000, 4200, 150)])

    dd = joblib.load(Path(__file__).parent.resolve().joinpath("dustin_exp.joblib"))
    dd = _remove_nan_values(dd)
    dd = jax.tree_map(jnp.asarray, dd)

    if q_inv:
        qrel = lambda q1, q2: maths.quat_mul(maths.quat_inv(q1), q2)
    else:
        qrel = lambda q1, q2: maths.quat_mul(q1, maths.quat_inv(q2))

    X = {
        "seg1": {"acc": dd["acc1"], "gyr": dd["gyr1"]},
        "seg3": {"acc": dd["acc3"], "gyr": dd["gyr3"]},
    }
    y = {
        "seg1": {"seg2": qrel(dd["q1"], dd["q2"]), "seg3": qrel(dd["q2"], dd["q3"])},
        "seg2": {"seg1": qrel(dd["q2"], dd["q1"]), "seg3": qrel(dd["q2"], dd["q3"])},
        "seg3": {"seg2": qrel(dd["q3"], dd["q2"]), "seg1": qrel(dd["q2"], dd["q1"])},
    }[anchor]

    data = X, y

    @jax.vmap
    def extract_windows(start):
        return jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_slice_in_dim(arr, start, T), data
        )

    return extract_windows(start_indices)


def dustin_exp_Xy_with_imus(
    anchor: str = "seg1", q_inv: bool = True
) -> Tuple[jax.Array, jax.Array]:
    start_indices = jnp.array([start for start in range(3000, 4200, 150)])

    dd = joblib.load(Path(__file__).parent.resolve().joinpath("dustin_exp.joblib"))
    dd = _remove_nan_values(dd)
    dd = jax.tree_map(jnp.asarray, dd)

    if q_inv:
        qrel = lambda q1, q2: maths.quat_mul(maths.quat_inv(q1), q2)
    else:
        qrel = lambda q1, q2: maths.quat_mul(q1, maths.quat_inv(q2))

    X = {
        "imu1": {"acc": dd["acc1"], "gyr": dd["gyr1"]},
        "imu2": {"acc": dd["acc3"], "gyr": dd["gyr3"]},
    }
    y = {
        "seg1": {"seg2": qrel(dd["q1"], dd["q2"]), "seg3": qrel(dd["q2"], dd["q3"])},
        "seg2": {"seg1": qrel(dd["q2"], dd["q1"]), "seg3": qrel(dd["q2"], dd["q3"])},
        "seg3": {"seg2": qrel(dd["q3"], dd["q2"]), "seg1": qrel(dd["q2"], dd["q1"])},
    }[anchor]

    y.update(
        {
            "imu1": maths.unit_quats_like(dd["q1"]),
            "imu2": maths.unit_quats_like(dd["q1"]),
        }
    )

    data = X, y

    @jax.vmap
    def extract_windows(start):
        return jax.tree_util.tree_map(
            lambda arr: jax.lax.dynamic_slice_in_dim(arr, start, T), data
        )

    return extract_windows(start_indices)
