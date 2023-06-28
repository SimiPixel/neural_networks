from collections import defaultdict
from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
from x_xy import base, scan
from x_xy.maths import safe_normalize, safe_normalize_custom_jvp


def rnno_v2_reverse(
    sys: base.System,
    state_dim: int = 400,
    message_dim: int = 200,
    standardize_message: bool = False,
    standardize_state: bool = False,
    state_init=jnp.zeros,
    message_init=jnp.zeros,
    custom_jvp: bool = False,
) -> SimpleNamespace:
    "Expects unbatched inputs. Batching via `vmap`"

    if custom_jvp:
        normalize = safe_normalize_custom_jvp
    else:
        normalize = safe_normalize

    @hk.without_apply_rng
    @hk.transform_with_state
    def timestep(X):
        recv_msg_from_top = hk.GRU(state_dim)
        recv_external = hk.GRU(state_dim)
        recv_msg_from_bot = hk.GRU(state_dim)
        send_msg_to_bot = hk.nets.MLP([state_dim, message_dim])
        send_msg_to_top = hk.nets.MLP([state_dim, message_dim])
        send_external = hk.nets.MLP([state_dim, 4])

        state = hk.get_state("state", [sys.num_links(), state_dim], init=state_init)
        empty_message = hk.get_state("empty_message", [message_dim], init=message_init)
        state = {i: state[i] for i in range(sys.num_links())}

        mailbox = defaultdict(lambda: empty_message)

        def scan_bot_to_top_recv(_, __, i: int, p: int, name: str):
            # all childs have left their messages in the mailbox
            local_state, _ = recv_msg_from_bot(mailbox[i], state[i])

            # recv imu data & update local state
            local_measurement = (
                jnp.concatenate((X[name]["acc"], X[name]["gyr"]))
                if name in X
                else jnp.zeros((6,))
            )
            local_state, _ = recv_external(local_measurement, local_state)

            if p != -1:
                # leave message in mailbox of parent
                local_message = send_msg_to_top(local_state)
                if standardize_message:
                    local_message = hk.LayerNorm(-1, False, False)(local_message)
                mailbox[p] = mailbox[p] + local_message

            # save local state
            if standardize_state:
                local_state = hk.LayerNorm(-1, False, False)(local_state)
            state[i] = local_state

        scan.tree(
            sys,
            scan_bot_to_top_recv,
            "lll",
            list(range(sys.num_links())),
            sys.link_parents,
            sys.link_names,
            reverse=True,
        )

        y = {}
        msg = {-1: empty_message}

        def scan_top_to_bot_send(_, __, i: int, p: int, name: str):
            # recv message from top & update local state
            local_state, _ = recv_msg_from_top(msg[p], state[i])

            if p != -1:
                # send orientation estimate to outside world
                y[name] = normalize(send_external(local_state))

            # send message to bot
            local_message = send_msg_to_bot(local_state)
            if standardize_message:
                local_message = hk.LayerNorm(-1, False, False)(local_message)
            msg[i] = local_message

            # save local state
            if standardize_state:
                local_state = hk.LayerNorm(-1, False, False)(local_state)
            state[i] = local_state

        scan.tree(
            sys,
            scan_top_to_bot_send,
            "lll",
            list(range(sys.num_links())),
            sys.link_parents,
            sys.link_names,
        )

        state = jnp.concatenate([state[i][None] for i in range(sys.num_links())])
        hk.set_state("state", state)

        return y

    def init(key, X):
        "Returns: (params, state)"
        X_at_t0 = jax.tree_map(lambda arr: arr[0], X)
        params, state = timestep.init(key, X_at_t0)
        return params, state

    def apply(params, state, X):
        "Returns: (y, state)"

        def swap_args(carry, X):
            y, carry = timestep.apply(params, carry, X)
            return carry, y

        def unrolled(state, X):
            return jax.lax.scan(swap_args, state, X)

        state_out, output = unrolled(state, X)
        return output, state_out

    return SimpleNamespace(init=init, apply=apply)
