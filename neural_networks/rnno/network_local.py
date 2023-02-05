from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils
from x_xy.maths import safe_normalize


def rnno_network_local(
    n_hidden_units: int = 400,
    message_dim: int = 200,
    length_of_chain: int = 3,
):
    """Graph Filter."""

    N = length_of_chain

    def get_local(state, node_nr: int):
        return jax.lax.dynamic_index_in_dim(state, node_nr, keepdims=False)

    def set_local(state, node_nr: int, local_new_state):
        return jax.lax.dynamic_update_index_in_dim(
            state, local_new_state, node_nr, axis=0
        )

    def local_measurement(X, node_nr):
        def left(X):
            return tree_utils.batch_concat(X[0], 0)

        def right(X):
            return tree_utils.batch_concat(X[N - 1], 0)

        def no_imu(X):
            return jnp.zeros((6,))

        return jax.lax.cond(
            jnp.isin(node_nr, 0),
            left,
            lambda X: jax.lax.cond(jnp.isin(node_nr, N - 1), right, no_imu, X),
            X,
        )

    @hk.without_apply_rng
    @hk.transform_with_state
    def scan_time(X):
        recv_msg_from_top = hk.GRU(n_hidden_units)
        recv_external = hk.GRU(n_hidden_units)
        recv_msg_from_bot = hk.GRU(n_hidden_units)
        send_msg_to_bot = hk.nets.MLP([n_hidden_units, message_dim])
        send_msg_to_top = hk.nets.MLP([n_hidden_units, message_dim])
        send_external = hk.nets.MLP([n_hidden_units, 4])

        state = hk.get_state("state", [N, n_hidden_units], init=jnp.zeros)

        def scan_top2bot(carry, _):
            recv_msg, state, node_nr = carry
            local_state = get_local(state, node_nr)
            local_state, _ = recv_msg_from_top(recv_msg, local_state)
            measurement = local_measurement(X, node_nr)
            local_state, _ = recv_external(measurement, local_state)
            state = set_local(state, node_nr, local_state)
            send_msg = send_msg_to_bot(local_state)
            assert send_msg.shape == (message_dim,)
            return (send_msg, state, node_nr + 1), _

        def scan_bot2top(carry, _):
            recv_msg, state, node_nr = carry
            local_state = get_local(state, node_nr)
            local_state, _ = recv_msg_from_bot(recv_msg, local_state)
            state = set_local(state, node_nr, local_state)
            y = send_external(local_state)
            send_msg = send_msg_to_top(local_state)
            assert send_msg.shape == (message_dim,)
            return (send_msg, state, node_nr - 1), y

        empty_msg = jnp.zeros((message_dim,))
        state = hk.scan(scan_top2bot, (empty_msg, state, 0), xs=None, length=N)[1]
        carry, ys = hk.scan(scan_bot2top, (empty_msg, state, N - 1), xs=None, length=N)
        state = carry[1]
        hk.set_state("state", state)

        # convert first axis to list; for tree_map
        qs = jax.tree_map(safe_normalize, [q for q in ys[1:]])

        # order is reversed due to top-bottom scan
        node_nrs = list(range(N - 1, 0, -1))
        return dict(zip(node_nrs, qs))

    def init(key, X):
        X_at_t0 = jax.tree_map(lambda arr: arr[0], X)
        params, state = scan_time.init(key, X_at_t0)
        return params, state

    def apply(params, state, X):
        def swap_args(carry, X):
            y, carry = scan_time.apply(params, carry, X)
            return carry, y

        def unrolled(state, X):
            return jax.lax.scan(swap_args, state, X)

        state_out, output = unrolled(state, X)
        return output, state_out

    return SimpleNamespace(init=init, apply=apply)
