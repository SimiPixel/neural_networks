import pickle
from pathlib import Path
from typing import Union

from tree_utils import PyTree

from neural_networks.rnno.train import Logger
from neural_networks.rnno.training_loop import TrainingLoopCallback

suffix = ".pickle"


def save(data: PyTree, path: Union[str, Path], overwrite: bool = False):
    path = Path(path).expanduser()
    if path.suffix != suffix:
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if overwrite:
            path.unlink()
        else:
            raise RuntimeError(f"File {path} already exists.")
    with open(path, "wb") as file:
        pickle.dump(data, file)


def load(path: Union[str, Path]) -> PyTree:
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    if path.suffix != suffix:
        raise ValueError(f"Not a {suffix} file: {path}")
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


class SaveParamsTrainingLoopCallback(TrainingLoopCallback):
    def __init__(self, n_episodes: int, path_to_file: str):
        self.n_episodes = n_episodes
        self.path_to_file = str(
            Path(path_to_file).expanduser().with_suffix("").with_suffix(".pickle")
        )

    def after_training_step(
        self,
        i_episode: int,
        metrices: dict,
        params: dict,
        sample_eval: dict,
        loggers: list[Logger],
    ) -> None:
        if i_episode != self.n_episodes - 1:
            return
        neptune_run = loggers[0].run
        # params is Lookahead object
        save(params.slow, self.path_to_file, overwrite=True)
        print(f"Uploading file {self.path_to_file}")
        neptune_run["final_params/path_to_file"].log(self.path_to_file)
        neptune_run["final_params/params.pickle"].upload(self.path_to_file)
