import jax
import x_xy

from neural_networks.rnno import train, wavenet


def finalize_fn_full_imu_setup(key, q, x, sys):
    X = {
        name: x_xy.algorithms.imu(x.take(sys.name_to_idx(name), 1), sys.gravity, sys.dt)
        for name in sys.link_names
    }
    return X


def finalize_fn_rel_pose(key, q, x, sys):
    return x_xy.algorithms.rel_pose(sys, x)


def finalize_fn(*args):
    X = finalize_fn_full_imu_setup(*args)
    y = finalize_fn_rel_pose(*args)
    return X, y


def test_rnno_v1():
    for i, example in enumerate(x_xy.io.list_examples()):
        # the `symmetric` example has only one body, i.e.
        # has no relative pose
        if (
            example == "symmetric"
            or example == "spherical_stiff"
            or example == "test_free"
        ):
            continue

        print("Example: ", example)
        sys = x_xy.io.load_example(example)
        seed = jax.random.PRNGKey(1)
        gen = x_xy.algorithms.build_generator(
            sys, x_xy.algorithms.RCMG_Config(T=10.0), finalize_fn=finalize_fn
        )
        gen = x_xy.algorithms.batch_generator(gen, 2)
        X, y = gen(seed)

        network = wavenet(sys, 1, 5, 8, 12)
        params, state = network.init(seed, X)
        y = network.apply(params, state, X)[0]

        for name in sys.link_names:
            assert name in X
            for sensor in ["acc", "gyr"]:
                assert sensor in X[name]
                assert X[name][sensor].shape == (2, 1000, 3)

            p = sys.link_parents[sys.name_to_idx(name)]
            if p == -1:
                assert name not in y
            else:
                assert name in y
                assert y[name].shape == (2, 1000, 4)

        # test that it is trainable
        train(gen, 3, network)
