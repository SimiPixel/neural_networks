import jax
import tqdm
import tree_utils

from neural_networks.logging import Logger, n_params


class TrainingLoopCallback:
    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        pass


class TrainingLoop:
    def __init__(
        self,
        key,
        generator,
        params,
        opt_state,
        step_fn,
        loggers: list[Logger],
        callbacks: list[TrainingLoopCallback] = [],
    ):
        self._key = key
        self.i_episode = -1
        self._generator = generator
        self._params = params
        self._opt_state = opt_state
        self._step_fn = step_fn
        self._loggers = loggers
        self._callbacks = callbacks

        self._sample_eval = generator(jax.random.PRNGKey(0))
        batchsize = tree_utils.tree_shape(self._sample_eval, 0)
        T = tree_utils.tree_shape(self._sample_eval, 1)

        for logger in loggers:
            logger.log(dict(n_params=n_params(params.fast), batchsize=batchsize, T=T))

    @property
    def key(self):
        self._key, consume = jax.random.split(self._key)
        return consume

    def run(self, n_episodes: int = 1, close_afterwards: bool = True):
        for _ in tqdm.tqdm(range(n_episodes)):
            self.step()

        if close_afterwards:
            self.close()

    def step(self):
        self.i_episode += 1

        sample_train = self._sample_eval
        self._sample_eval = self._generator(self.key)

        self._params, self._opt_state, loss = self._step_fn(
            self._params, self._opt_state, sample_train["X"], sample_train["y"]
        )

        metrices = {}
        metrices.update(loss)

        for callback in self._callbacks:
            callback.after_training_step(
                self.i_episode, metrices, self._params, self._sample_eval, self._loggers
            )

        for logger in self._loggers:
            logger.log(metrices)

        return metrices

    def close(self):
        for logger in self._loggers:
            logger.close()
