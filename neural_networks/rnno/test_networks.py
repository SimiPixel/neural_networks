import jax
import jax.numpy as jnp
import x_xy

from neural_networks.rnno.network_local import rnno_network


def make_train_data(key, q, x, sys):
    X = {
        name: x_xy.algorithms.imu(x.take(sys.name_to_idx(name), 1), sys.gravity, sys.dt)
        for name in sys.link_names
    }
    return X


def test_rnno():
    for i, example in enumerate(x_xy.io.list_examples()):
        sys = x_xy.io.load_example(example)
        seed = jax.random.PRNGKey(1)
        X = x_xy.algorithms.build_generator(sys, finalize_fn=make_train_data)(seed)
        rnno = rnno_network(sys, 40, 20)

        if i == 0:
            params, state = rnno.init(seed, X)
        else:
            _, state = rnno.init(seed, X)

        y = rnno.apply(params, state, X)[0]
