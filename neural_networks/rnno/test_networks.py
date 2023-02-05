import jax
import jax.numpy as jnp

from neural_networks.rnno.network import rnno_network
from neural_networks.rnno.network_local import rnno_network_local


def test_networks():
    # this should be the output of a `x_xy` simulation
    Ts = 0.01
    for T in [10, 20]:
        N = int(T / Ts)
        for network_fn in [rnno_network, rnno_network_local]:
            for length_of_chain in [3, 4, 5]:
                X = {
                    0: {"acc": jnp.ones((N, 3)), "gyr": jnp.ones((N, 3))},
                    length_of_chain - 1: jnp.ones((N, 6)),
                }
                network = network_fn(length_of_chain=length_of_chain)
                params, state = network.init(jax.random.PRNGKey(1), X)
                y, state = network.apply(params, state, X)
                for i in range(1, length_of_chain):
                    assert y[i].shape == (N, 4)
