from collections import defaultdict
from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
from x_xy import base, scan
from x_xy.maths import safe_normalize


def rnno_v2(
    sys: base.System,
    state_dim: int = 400,
    message_dim: int = 200,
    layernorm: bool = False,
) -> SimpleNamespace:
    "Expects unbatched inputs. Batching via `vmap`"

    @hk.without_apply_rng
    @hk.transform_with_state
    def timestep(X):
        prefix_layernorm = lambda cell: hk.Sequential(
            [hk.LayerNorm(-1, False, False)] if layernorm else [] + [cell]
        )

        recv_msg_from_top = prefix_layernorm(hk.GRU(state_dim))
        recv_external = prefix_layernorm(hk.GRU(state_dim))
        recv_msg_from_bot = prefix_layernorm(hk.GRU(state_dim))
        send_msg_to_bot = hk.nets.MLP([state_dim, message_dim])
        send_msg_to_top = hk.nets.MLP([state_dim, message_dim])
        send_external = hk.nets.MLP([state_dim, 4])

        state = hk.get_state("state", [sys.num_links(), state_dim], init=jnp.zeros)

        state = {i: state[i] for i in range(sys.num_links())}
        msg = {-1: jnp.zeros((message_dim,))}

        def scan_top_to_bot_recv_imu_data(_, __, i: int, p: int, name: str):
            # recv message from top & update local state
            local_state, _ = recv_msg_from_top(msg[p], state[i])

            # recv imu data & update local state
            local_measurement = (
                jnp.concatenate((X[name]["acc"], X[name]["gyr"]))
                if name in X
                else jnp.zeros((6,))
            )
            local_state, _ = recv_external(local_measurement, local_state)

            # send message to bot
            msg[i] = send_msg_to_bot(local_state)

            # save local state
            state[i] = local_state

        scan.tree(
            sys,
            scan_top_to_bot_recv_imu_data,
            "lll",
            list(range(sys.num_links())),
            sys.link_parents,
            sys.link_names,
        )

        y = {}
        mailbox = defaultdict(lambda: jnp.zeros((message_dim,)))

        def scan_bot_to_top_send_ori(_, __, i: int, p: int, name: str):
            # all childs have left their messages in the mailbox
            local_state, _ = recv_msg_from_bot(mailbox[i], state[i])
            state[i] = local_state

            if p == -1:
                return

            # send orientation estimate to outside world
            y[name] = safe_normalize(send_external(local_state))

            # leave message in mailbox of parent
            mailbox[p] = mailbox[p] + send_msg_to_top(local_state)

        scan.tree(
            sys,
            scan_bot_to_top_send_ori,
            "lll",
            list(range(sys.num_links())),
            sys.link_parents,
            sys.link_names,
            reverse=True,
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
