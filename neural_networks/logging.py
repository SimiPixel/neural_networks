import logging
import os
import time
from abc import ABC, abstractmethod, abstractstaticmethod
from typing import Optional

import jax
import neptune
import numpy as np
import wandb
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


class DictLogger(Logger):
    def __init__(self, print_on_close: bool = False):
        self._logs = {}
        self._print_on_close = print_on_close

    def log(self, metrics: NestedDict):
        metrics = _flatten_convert_filter_nested_dict(metrics, filter_nan_inf=False)
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


class MultimediaLogger(Logger):
    @abstractmethod
    def log_image(self, path: str):
        pass

    @abstractmethod
    def log_video(self, path: str):
        pass

    @abstractmethod
    def log_params(self, path: str):
        pass

    def log(self, metrics: NestedDict):
        for key, value in _flatten_convert_filter_nested_dict(metrics):
            self.log_key_value(key, value)

    @abstractmethod
    def log_key_value(self, key: str, value: str | float):
        pass

    def log_command_output(self, command: str):
        path = command.replace(" ", "_") + ".txt"
        os.system(f"{command} >> {path}")
        self.log_txt(path, wait=True)
        os.system(f"rm {path}")

    @abstractmethod
    def log_txt(self, path: str, wait: bool = True):
        pass

    @staticmethod
    def _print_upload_file(path: str):
        logging.info(f"Uploading file {path}.")

    @abstractstaticmethod
    def disable():
        pass


class NeptuneLogger(MultimediaLogger):
    def __init__(
        self,
        project: Optional[str] = None,
        name: Optional[str] = None,
    ):
        """Logger that logs the training progress to Neptune.

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

        if name is None:
            name = os.environ.get("NEPTUNE_NAME", None)

        self.run = neptune.init_run(
            name=name,
            project=project,
            api_token=api_token,
        )

        _log_environment(self)

    def log_key_value(self, key: str, value: str | float):
        self.run[key] = value

    def log_params(self, path: str):
        self._print_upload_file(path)
        # if we wouldn't wait then the run might end before upload finishes
        self.run[f"params/{_file_name(path, extension=True)}"].upload(path, wait=True)

    def log_video(self, path: str):
        self.run[f"video/{_file_name(path, extension=True)}"].upload(path)

    def log_image(self, path: str):
        self.run[f"image/{_file_name(path, extension=True)}"].upload(path)

    def log_txt(self, path: str, wait: bool = True):
        self.run[f"txt/{_file_name(path)}"].upload(path, wait=wait)

    def close(self):
        self.run.stop()

    @staticmethod
    def disable():
        os.environ["NEPTUNE_MODE"] = "debug"


class WandbLogger(MultimediaLogger):
    def __init__(self):
        _log_environment(self)

    def log_key_value(self, key: str, value: str | float):
        wandb.log({key: value})

    def log_params(self, path: str):
        self._print_upload_file(path)
        wandb.save(path, policy="now")

    def log_video(self, path: str, fps: int = 25, caption: Optional[str] = None):
        wandb.log({"video": wandb.Video(path, caption=caption, fps=fps)})

    def log_image(self, path: str, caption: Optional[str] = None):
        wandb.log({"image": wandb.Image(path, caption=caption)})

    def log_txt(self, path: str, wait: bool = True):
        wandb.save(path, policy="now")
        # TODO: `wandb` is not async at all?
        if wait:
            time.sleep(3)

    @staticmethod
    def disable():
        os.environ["WANDB_MODE"] = "offline"


def disable_syncing_to_cloud():
    NeptuneLogger.disable()
    WandbLogger.disable()


def _file_name(path: str, extension: bool = False):
    file = path.split("/")[-1]
    return file if extension else file.split(".")[0]


def _log_environment(logger: MultimediaLogger):
    logger.log_command_output("pip list")
    logger.log_command_output("conda list")
    logger.log_command_output("nvidia-smi")


def _flatten_convert_filter_nested_dict(
    metrices: NestedDict, filter_nan_inf: bool = True
):
    metrices = _flatten_dict(metrices)
    metrices = jax.tree_map(to_float_if_not_string, metrices)

    if not filter_nan_inf:
        return metrices

    filtered_metrices = {}
    for key, value in metrices.items():
        if not isinstance(value, str) and (np.isnan(value) or np.isinf(value)):
            print(f"Warning: Value of metric {key} is {value}. We skip it.")
            continue
        filtered_metrices[key] = value
    return filtered_metrices


def _flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        k = str(k) if isinstance(k, int) else k
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def to_float_if_not_string(value):
    if isinstance(value, str):
        return value
    else:
        return float(value)
