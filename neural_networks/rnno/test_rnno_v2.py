import jax
import x_xy

from neural_networks.rnno import (
    LogGradsTrainingLoopCallBack,
    dustin_exp_xml,
    rnno_v2,
    train,
)


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


def test_rnno_v2():
    rnno_dustin = rnno_v2(x_xy.io.load_sys_from_str(dustin_exp_xml), 40, 20)

    for i, example in enumerate(x_xy.io.list_examples()):
        print("Example: ", example)
        sys = x_xy.io.load_example(example)
        seed = jax.random.PRNGKey(1)
        gen = x_xy.algorithms.build_generator(
            sys, x_xy.algorithms.RCMG_Config(T=10.0), finalize_fn=finalize_fn
        )
        gen = x_xy.algorithms.batch_generator([gen, gen], [16, 16])

        X, y = gen(seed)
        X, y = jax.tree_map(lambda arr: arr[0], (X, y))

        rnno = rnno_v2(sys, 40, 20)

        if i == 0:
            params, state = rnno.init(seed, X)
        else:
            # `params` are universal across systems
            _, state = rnno.init(seed, X)

        y = rnno.apply(params, state, X)[0]

        for name in sys.link_names:
            assert name in X
            for sensor in ["acc", "gyr"]:
                assert sensor in X[name]
                assert X[name][sensor].shape == (1000, 3)

            p = sys.link_parents[sys.name_to_idx(name)]
            if p == -1:
                assert name not in y
            else:
                assert name in y
                assert y[name].shape == (1000, 4)

        # the `symmetric` example has only one body, i.e.
        # has no relative pose
        if example == "symmetric" or example == "spherical_stiff" or example == "free":
            continue

        train(
            gen,
            5,
            rnno,
            network_dustin=rnno_dustin,
            callbacks=[LogGradsTrainingLoopCallBack()],
        )


def test_layernorm():
    rnno_dustin = rnno_v2(x_xy.io.load_sys_from_str(dustin_exp_xml), 40, 20, True)

    # one is enough
    N = 1

    for i, example in enumerate(x_xy.io.list_examples()[:N]):
        print("Example: ", example)
        sys = x_xy.io.load_example(example)
        seed = jax.random.PRNGKey(1)
        gen = x_xy.algorithms.build_generator(
            sys, x_xy.algorithms.RCMG_Config(T=10.0), finalize_fn=finalize_fn
        )
        gen = x_xy.algorithms.batch_generator([gen, gen], [16, 16])

        X, y = gen(seed)
        X, y = jax.tree_map(lambda arr: arr[0], (X, y))

        rnno = rnno_v2(sys, 40, 20, True)

        if i == 0:
            params, state = rnno.init(seed, X)
        else:
            # `params` are universal across systems
            _, state = rnno.init(seed, X)

        y = rnno.apply(params, state, X)[0]

        for name in sys.link_names:
            assert name in X
            for sensor in ["acc", "gyr"]:
                assert sensor in X[name]
                assert X[name][sensor].shape == (1000, 3)

            p = sys.link_parents[sys.name_to_idx(name)]
            if p == -1:
                assert name not in y
            else:
                assert name in y
                assert y[name].shape == (1000, 4)

        # the `symmetric` example has only one body, i.e.
        # has no relative pose
        if example == "symmetric" or example == "spherical_stiff" or example == "free":
            continue

        train(gen, 5, rnno, network_dustin=rnno_dustin)
