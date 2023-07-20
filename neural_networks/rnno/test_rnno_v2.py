import jax
import numpy as np
import x_xy
from x_xy.subpkgs import pipeline

from neural_networks.rnno import (
    LogGradsTrainingLoopCallBack,
    dustin_exp_xml,
    rnno_v2,
    rnno_v2_dw,
    rnno_v2_minimal,
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


def test_vmap_version_is_equal():
    example = "test_morph_system/four_seg_seg1"
    sys = x_xy.io.load_example(example)
    seed = jax.random.PRNGKey(1)
    gen = x_xy.algorithms.build_generator(
        sys, x_xy.algorithms.RCMG_Config(T=10.0), finalize_fn=finalize_fn
    )
    X, _ = gen(seed)

    def yhat(vmap_version: bool):
        return pipeline.predict(
            sys, lambda sys: rnno_v2_minimal(sys, 40, 20, vmap_version=vmap_version), X
        )[0]

    jax.tree_map(
        lambda a, b: np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-6),
        yhat(False),
        yhat(True),
    )


def test_rnno_v2():
    for rnno_fn in [rnno_v2_dw, rnno_v2_minimal]:
        rnno_dustin = rnno_fn(x_xy.io.load_sys_from_str(dustin_exp_xml), 10, 2)

        for i, example in enumerate(x_xy.io.list_examples()):
            print("Example: ", example)
            sys = x_xy.io.load_example(example)
            seed = jax.random.PRNGKey(1)
            gen = x_xy.algorithms.build_generator(
                sys, x_xy.algorithms.RCMG_Config(T=10.0), finalize_fn=finalize_fn
            )
            gen = x_xy.algorithms.batch_generator([gen, gen], [1, 1])

            X, y = gen(seed)
            X, y = jax.tree_map(lambda arr: arr[0], (X, y))

            rnno = rnno_fn(sys, 10, 2)

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
            if (
                example == "symmetric"
                or example == "spherical_stiff"
                or example == "free"
            ):
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
