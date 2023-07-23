import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional

import jax
import neptune
import numpy as np
from tree_utils import PyTree, tree_batch

# An arbitrarily nested dictionary with jax.Array leaves; Or strings
NestedDict = PyTree


class Logger(ABC):
    @abstractmethod
    def log(self, metrics: NestedDict):
        pass

    def close(self):
        pass


def n_params(params):
    return sum([arr.flatten().size for arr in jax.tree_util.tree_leaves(params)])


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        k = str(k) if isinstance(k, int) else k
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def to_float_if_not_string(value):
    if isinstance(value, str):
        return value
    else:
        return float(value)


class DictLogger(Logger):
    def __init__(self, print_on_close: bool = False):
        self._logs = {}
        self._print_on_close = print_on_close

    def log(self, metrics: NestedDict):
        metrics = flatten_dict(metrics)
        metrics = jax.tree_map(to_float_if_not_string, metrics)

        metrics = tree_batch([metrics])

        for key in metrics:
            existing_keys = []
            if key in self._logs:
                existing_keys.append(key)
            else:
                self._logs[key] = metrics[key]

        if len(existing_keys) > 0:
            self._logs.update(
                tree_batch(
                    [
                        {key: self._logs[key] for key in existing_keys},
                        {key: metrics[key] for key in existing_keys},
                    ],
                    True,
                )
            )

    def close(self):
        if self._print_on_close:
            print(self._logs)


class NeptuneLogger(Logger):
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
        force_logging: bool = False,
    ):
        """Logger that logs the training progress to Neptune.
        Does not log if `NEPTUNE_DISABLE` is set to `1`, unless `force_logging` is set,
        then it allways logs.

        Args:
            project (Optional[str], optional): Name of the project where the run should
                go, in the form "workspace-name/project_name". Can also be provided
                using the environemnt variable `NEPTUNE_PROJECT`
            name (Optional[str], optional): Identifier inside the project. Can also be
            provided using the environment variable `NEPTUNE_NAME`

        Raises:
            Exception: If environment variable `NEPTUNE_TOKEN` is unset.
        """
        api_token = os.environ.get("NEPTUNE_TOKEN", None)
        if api_token is None:
            raise Exception(
                "Could not find the token for neptune logging. Make sure that the \
                            environment variable `NEPTUNE_TOKEN` is set."
            )

        self._stop_logging = bool(os.environ.get("NEPTUNE_DISABLE", None))

        # overwrite if flag is set
        if force_logging:
            self._stop_logging = False

        if self._stop_logging:
            return

        if name is None:
            name = os.environ.get("NEPTUNE_NAME", None)

        self.run = neptune.init_run(
            name=name,
            project=project,
            api_token=api_token,
        )

        # Record exact start of training
        self.run["train/start"] = datetime.now()

        # Record all package versions in conda env
        for manager in ["pip", "conda"]:
            # First, dump into a txt-file
            os.system(f'{manager} list >> "{manager}_list.txt"')
            # Then upload
            # flag `wait`=True is required, otherwise it won't work
            self.run[f"package_versions/{manager}"].upload(
                f"{manager}_list.txt", wait=True
            )
            # Then remove
            os.system(f"rm {manager}_list.txt")

        # Log nvidia-smi
        os.system("nvidia-smi >> nvidia_smi.txt")
        self.run["nvidia_smi"].upload("nvidia_smi.txt", wait=True)
        os.system("rm nvidia_smi.txt")

    def log(self, metrices) -> None:
        if self._stop_logging:
            return

        metrices = flatten_dict(metrices)
        metrices = jax.tree_map(to_float_if_not_string, metrices)

        for key, value in metrices.items():
            if not isinstance(value, str) and np.isnan(value):
                print(f"Warning: Value of metric {key} is {value}. We skip it.")
                continue

            if not isinstance(value, str) and np.isinf(value):
                print(f"Warning: Value of metric {key} is {value}. We skip it.")
                continue

            self.run[key].log(value)

    def log_params(self, path_to_file: str) -> None:
        if self._stop_logging:
            return

        print(f"Uploading file {path_to_file}")
        self.run["final_params/path_to_file"].log(path_to_file)
        self.run["final_params/params.pickle"].upload(path_to_file)

    def log_video(self, path_to_file: str):
        if self._stop_logging:
            return

        self.run[f"video/{_file_name(path_to_file)}"].upload(path_to_file)

    def log_image(self, path_to_file: str):
        if self._stop_logging:
            return

        self.run[f"image/{_file_name(path_to_file)}"].upload(path_to_file)

    def close(self):
        if not self._stop_logging:
            # Record exact end of training
            self.run["train/end"] = datetime.now()

        return super().close()


def _file_name(path: str):
    return path.split("/")[-1].split(".")[0]
