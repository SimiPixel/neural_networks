import os
from datetime import datetime

import jax
import jax.numpy as jnp
import optax
import pytest

from neural_networks.io_params import load, save
from neural_networks.logging import NeptuneLogger
from neural_networks.rnno import SaveParamsTrainingLoopCallback
from neural_networks.rnno.training_loop import TrainingLoop


def test_save_load():
    params = {"matrix": jnp.zeros((100, 100))}
    test_file = "~/params1/params.pickle"
    save(params, test_file, True)
    save(params, test_file, True)
    with pytest.raises(RuntimeError):
        save(params, test_file, overwrite=False)
    load(test_file)

    # clean up
    os.system("rm ~/params1/params.pickle")
    os.system("rmdir ~/params1")


def generator(key):
    # time, features
    X = y = jnp.zeros(
        (
            1,
            1,
        )
    )
    return X, y


def step_fn(params, opt_state, X, y):
    import time

    time.sleep(0.02)
    debug_grads = [params, params]
    return params, opt_state, {"loss": jnp.array(0.0)}, debug_grads


def test_save_params_loop_callback():
    params = {"matrix": jnp.zeros((100, 100))}
    params = optax.LookaheadParams(params, params)
    test_file = "~/params2/params.pickle"
    logger = NeptuneLogger("iss/test", name=str(datetime.now()))
    n_episodes = 100
    callback = SaveParamsTrainingLoopCallback(test_file)

    opt_state = None
    loop = TrainingLoop(
        jax.random.PRNGKey(1),
        generator,
        params,
        opt_state,
        step_fn,
        [logger],
        [callback],
    )
    loop.run(n_episodes)

    import time

    # await upload
    time.sleep(3)
    # clean up
    os.system("rm ~/params2/params.pickle")
    os.system("rmdir ~/params2")


if __name__ == "__main__":
    test_save_params_loop_callback()
