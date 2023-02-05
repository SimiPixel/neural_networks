import haiku as hk
import jax
import jax.numpy as jnp
import tree_utils
from x_xy.maths import safe_normalize

more_more_complex = dict(
    rnn_layers=(500, 400), linear_layers=(250, 150, 75, 75, 50, 50)
)
more_complex = dict(rnn_layers=(400, 300), linear_layers=(200, 100, 50, 50, 25, 25))
complex = dict(rnn_layers=(300, 200), linear_layers=(100, 50, 50, 25, 25))
medium = dict(rnn_layers=(100, 100), linear_layers=(50, 25))
shallow = dict(rnn_layers=(100,), linear_layers=(25,))
complexities = dict(
    more_complex=more_more_complex, complex=more_complex, medium=complex, shallow=medium
)


def rnno_network(
    rnn_layers=(100,),
    rnn_cell=hk.GRU,
    linear_layers=(),
    layernorm=True,
    act_fn_linear=jax.nn.relu,
    act_fn_rnn=jax.nn.elu,
    length_of_chain=3,
):
    """RNN-neural net.
    (time, features) -> (time, 4*n_output_quats), norm_mse
    """
    N = length_of_chain

    @hk.without_apply_rng
    @hk.transform_with_state
    def forward_fn(X):
        # extract measurements
        X = tree_utils.batch_concat((X[0], X[N - 1]))

        for i, n_units in enumerate(rnn_layers):
            state = hk.get_state(f"rnn_{i}", shape=[n_units], init=jnp.zeros)
            X, state = hk.dynamic_unroll(rnn_cell(n_units), X, state)
            hk.set_state(f"rnn_{i}", state)
            # layer-norm
            if layernorm:
                X = hk.LayerNorm(axis=-1, create_scale=False, create_offset=False)(X)
            X = act_fn_rnn(X)

        for n_units in linear_layers:
            X = hk.Linear(n_units)(X)
            X = act_fn_linear(X)

        out_dim = (N - 1) * 4
        X = hk.Linear(out_dim)(X)

        qs = jax.tree_map(
            safe_normalize, [X[:, i * 4 : (i + 1) * 4] for i in range(N - 1)]
        )
        node_nrs = list(range(1, N))
        return dict(zip(node_nrs, qs))

    return forward_fn
