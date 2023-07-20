from types import SimpleNamespace

import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils
from x_xy import base, scan
from x_xy.maths import safe_normalize


def _tree(sys, f, reverse: bool = False):
    return scan.tree(
        sys,
        f,
        "lll",
        list(range(sys.num_links())),
        sys.link_parents,
        sys.link_names,
        reverse=reverse,
    )


def rnno_v2_dw(
    sys: base.System,
    state_dim: int = 400,
    number_of_stacked_gru_cells: int = 1,
) -> SimpleNamespace:
    "Expects unbatched inputs. Batching via `vmap`"

    parent_array = jnp.array(sys.link_parents, dtype=jnp.int32)

    @hk.without_apply_rng
    @hk.transform_with_state
    def timestep(X):
        grus = [hk.GRU(state_dim) for _ in range(number_of_stacked_gru_cells)]
        mlp = hk.nets.MLP([state_dim, 4])

        state = hk.get_state(
            "state",
            [sys.num_links(), number_of_stacked_gru_cells, state_dim],
            init=jnp.zeros,
        )
        # each node sends the state of the last gru as message
        # root node sends an empty state as message
        empty_state = jnp.zeros(
            (
                1,
                state_dim,
            )
        )
        msg = jnp.concatenate((state[:, -1], empty_state))

        links = jnp.arange(sys.num_links())

        def populate_mailbox(link):
            return jnp.sum(
                jnp.where(
                    jnp.repeat((parent_array == link)[:, None], state_dim, axis=-1),
                    msg[:-1],
                    jnp.zeros((sys.num_links(), state_dim)),
                ),
                axis=0,
            )

        mailbox = jax.vmap(populate_mailbox)(links)

        def cell_input(_, __, i: int, p: int, name: str):
            local_measurement = (
                jnp.concatenate((X[name]["acc"], X[name]["gyr"]))
                if name in X
                else jnp.zeros((6,))
            )
            local_cell_input = tree_utils.batch_concat(
                (local_measurement, msg[p], mailbox[i]), num_batch_dims=0
            )
            return local_cell_input

        stacked_cell_input = _tree(sys, cell_input)

        def update_state(cell_input, state):
            output = cell_input
            next_state = []
            for i in range(number_of_stacked_gru_cells):
                output, next_state_i = grus[i](output, state[i])
                next_state.append(next_state_i)
            return safe_normalize(mlp(output)), jnp.stack(next_state)

        y, state = jax.vmap(update_state)(stacked_cell_input, state)
        hk.set_state("state", state)

        return {
            sys.idx_to_name(i): y[i]
            for i in range(sys.num_links())
            if sys.link_parents[i] != -1
        }

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
