from typing import Optional

import haiku as hk
import jax
import jax.numpy as jnp


def add_batch(nest, batch_size: Optional[int]):
    """Adds a batch dimension at axis 0 to the leaves of a nested structure."""
    broadcast = lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape)
    return jax.tree_util.tree_map(broadcast, nest)


class MGU(hk.RNNCore):
    def __init__(
        self,
        hidden_size: int,
        w_i_init: Optional[hk.initializers.Initializer] = None,
        w_h_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name=name)
        self.hidden_size = hidden_size
        self.w_i_init = w_i_init or hk.initializers.VarianceScaling()
        self.w_h_init = w_h_init or hk.initializers.VarianceScaling()
        self.b_init = b_init or jnp.zeros

    def __call__(self, inputs, state):
        if inputs.ndim not in (1, 2):
            raise ValueError("MGU input must be rank-1 or rank-2.")

        input_size = inputs.shape[-1]
        hidden_size = self.hidden_size
        w_i = hk.get_parameter(
            "w_i", [input_size, 2 * hidden_size], inputs.dtype, init=self.w_i_init
        )
        w_h = hk.get_parameter(
            "w_h", [hidden_size, 2 * hidden_size], inputs.dtype, init=self.w_h_init
        )
        b = hk.get_parameter("b", [2 * hidden_size], inputs.dtype, init=self.b_init)
        w_h_z, w_h_a = jnp.split(w_h, indices_or_sections=[hidden_size], axis=1)
        b_z, b_a = jnp.split(b, indices_or_sections=[hidden_size], axis=0)

        gates_x = jnp.matmul(inputs, w_i)
        z_x, a_x = jnp.split(gates_x, indices_or_sections=[hidden_size], axis=-1)

        z = jax.nn.sigmoid(z_x + jnp.matmul(state, w_h_z) + b_z)
        a = jnp.tanh(a_x + jnp.matmul(z * state, w_h_a) + b_a)
        next_state = (1 - z) * state + z * a
        return next_state, next_state

    def initial_state(self, batch_size: Optional[int]):
        state = jnp.zeros([self.hidden_size])
        if batch_size is not None:
            state = add_batch(state, batch_size)
        return state
