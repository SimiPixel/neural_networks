from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils
from x_xy import base, scan
from x_xy.maths import safe_normalize


def _tree(sys, f):
    return scan.tree(
        sys,
        f,
        "lll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_names,
    )


def _make_rnno_cell_apply_fn(sys, inner_cell, send_msg, send_quat, message_dim):
    parent_array = jnp.array(sys.link_parents, dtype=jnp.int32)

    def _rnno_cell_apply_fn(inputs, state):
        empty_message = jnp.zeros((1, message_dim))
        mailbox = jnp.repeat(empty_message, sys.num_links(), axis=0)
        state_flat = tree_utils.batch_concat_acme(state)
        msg = jnp.concatenate((jax.vmap(send_msg)(state_flat), empty_message))
        del state_flat

        def accumulate_message(link):
            return jnp.sum(
                jnp.where(
                    jnp.repeat((parent_array == link)[:, None], message_dim, axis=-1),
                    msg[:-1],
                    mailbox,
                ),
                axis=0,
            )

        mailbox = jax.vmap(accumulate_message)(jnp.arange(sys.num_links()))

        def cell_input(_, __, i: int, p: int, name: str):
            local_measurement = (
                jnp.concatenate((inputs[name]["acc"], inputs[name]["gyr"]))
                if name in inputs
                else jnp.zeros((6,))
            )
            local_cell_input = tree_utils.batch_concat(
                (local_measurement, msg[p], mailbox[i]), num_batch_dims=0
            )
            return local_cell_input

        stacked_cell_input = _tree(sys, cell_input)

        def update_state(cell_input, state):
            output, state = inner_cell(cell_input, state)
            return safe_normalize(send_quat(output)), state

        y, state = jax.vmap(update_state)(stacked_cell_input, state)

        outputs = {
            sys.idx_to_name(i): y[i]
            for i in range(sys.num_links())
            if sys.link_parents[i] != -1
        }
        return outputs, state

    return _rnno_cell_apply_fn


def rnno_v2_lstm(
    sys: base.System,
    hidden_state_dim: int = 400,
    message_dim: int = 200,
) -> SimpleNamespace:
    "Expects unbatched inputs. Batching via `vmap`"

    @hk.without_apply_rng
    @hk.transform_with_state
    def forward(X):
        inner_cell = hk.LSTM(hidden_state_dim)
        send_msg = hk.nets.MLP([hidden_state_dim, message_dim])
        send_quat = hk.nets.MLP([hidden_state_dim, 4])

        state_h_and_c = hk.get_state(
            "lstm_state", [sys.num_links(), 2 * hidden_state_dim], init=jnp.zeros
        )
        state = hk.LSTMState(
            state_h_and_c[:, :hidden_state_dim], state_h_and_c[:, hidden_state_dim:]
        )
        y, state = hk.dynamic_unroll(
            _make_rnno_cell_apply_fn(sys, inner_cell, send_msg, send_quat, message_dim),
            X,
            state,
        )
        hk.set_state("lstm_state", jnp.concatenate((state.hidden, state.cell), axis=1))
        return y

    return forward
