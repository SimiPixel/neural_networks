from typing import Sequence

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils
import x_xy
from x_xy.maths import safe_normalize


def rnno_v1(
    sys: x_xy.base.System,
    rnn_layers: Sequence[int] = (400, 300),
    linear_layers: Sequence[int] = (200, 100, 50, 50, 25, 25),
    layernorm: bool = True,
    act_fn_linear=jax.nn.relu,
    act_fn_rnn=jax.nn.elu,
):
    @hk.without_apply_rng
    @hk.transform_with_state
    def forward_fn(X):
        X = tree_utils.batch_concat(X)

        for i, n_units in enumerate(rnn_layers):
            state = hk.get_state(f"rnn_{i}", shape=[n_units], init=jnp.zeros)
            X, state = hk.dynamic_unroll(hk.GRU(n_units), X, state)
            hk.set_state(f"rnn_{i}", state)

            if layernorm:
                X = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(X)
            X = act_fn_rnn(X)

        for n_units in linear_layers:
            X = hk.Linear(n_units)(X)
            X = act_fn_linear(X)

        out_dim = _num_links_parent_not_root(sys.link_parents) * 4
        X = hk.Linear(out_dim)(X)

        quats = {}
        idx = 0

        def build_quaternion_output(_, __, name: str, p: int):
            nonlocal idx
            if p == -1:
                return
            quats[name] = safe_normalize(X[:, idx * 4 : (idx + 1) * 4])
            idx += 1

        x_xy.scan.tree(
            sys,
            build_quaternion_output,
            "ll",
            sys.link_names,
            sys.link_parents,
        )
        assert idx * 4 == X.shape[1]
        return quats

    return forward_fn


def _num_links_parent_not_root(parent_array: list[int]):
    return len([p for p in parent_array if p != -1])
