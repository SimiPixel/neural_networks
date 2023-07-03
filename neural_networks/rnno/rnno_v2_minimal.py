from collections import defaultdict
from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils
from x_xy import base, scan
from x_xy.maths import safe_normalize

from neural_networks.rnno.mgu import MGU


def _tree(sys, f, reverse: bool = False):
    scan.tree(
        sys,
        f,
        "lll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_names,
        reverse=reverse,
    )


class MLP(hk.Module):
    def __init__(
        self, layers, final_act_fn=None, stop_grads=False, name: str | None = None
    ):
        super().__init__(name)
        self._before = (
            (lambda x: jax.lax.stop_gradient(x)) if stop_grads else lambda x: x
        )
        self._mlp = hk.nets.MLP(layers)
        self._after = final_act_fn if final_act_fn else lambda x: x

    def __call__(self, x):
        return self._after(self._mlp(self._before(x)))


def rnno_v2_minimal(
    sys: base.System,
    state_dim: int = 400,
    message_dim: int = 200,
    state_init=jnp.zeros,
    message_init=jnp.zeros,
    use_mgu: bool = False,
    message_stop_grads: bool = False,
    message_tanh: bool = False,
    quat_tanh: bool = False,
) -> SimpleNamespace:
    "Expects unbatched inputs. Batching via `vmap`"

    cell = hk.GRU
    if use_mgu:
        cell = MGU

    @hk.without_apply_rng
    @hk.transform_with_state
    def timestep(X):
        recv_cell = cell(state_dim)
        send_msg = MLP(
            [state_dim, message_dim],
            jnp.tanh if message_tanh else None,
            message_stop_grads,
        )
        send_external = MLP([state_dim, 4], jnp.tanh if quat_tanh else None)

        state = hk.get_state("state", [sys.num_links(), state_dim], init=state_init)
        empty_message = hk.get_state("empty_message", [message_dim], init=message_init)

        state = {i: state[i] for i in range(sys.num_links())}
        msg = {-1: empty_message}

        # Step 1a): Pass messages to leaves
        def compute_messages(_, __, i: int, p: int, name: str):
            msg[i] = send_msg(state[i])

        _tree(sys, compute_messages)

        # Step 1b) Pass messages to root
        mailbox = defaultdict(lambda: empty_message)

        def compute_mailbox(_, __, i: int, p: int, name: str):
            if p != -1:
                mailbox[p] = mailbox[p] + send_msg(state[i])

        _tree(sys, compute_mailbox, reverse=True)

        # Step 2) Update node states & compute quaternion
        y = {}

        def update_state(_, __, i: int, p: int, name: str):
            local_measurement = (
                jnp.concatenate((X[name]["acc"], X[name]["gyr"]))
                if name in X
                else jnp.zeros((6,))
            )
            local_cell_input = tree_utils.batch_concat(
                (local_measurement, msg[p], mailbox[i]), num_batch_dims=0
            )
            output, state[i] = recv_cell(local_cell_input, state[i])
            if p != -1:
                y[name] = safe_normalize(send_external(output))

        _tree(sys, update_state)

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