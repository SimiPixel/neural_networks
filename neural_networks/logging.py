import os
from abc import ABC, abstractmethod
from typing import Optional

import jax
import neptune.new as neptune
from tree_utils import PyTree

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


class NeptuneLogger(Logger):
    def __init__(self, project: Optional[str] = None, name: Optional[str] = None):
        """Logger that logs the training progress to Neptune.
        Does not log if `NEPTUNE_DISABLE` is set to `1`.

        Args:
            project (Optional[str], optional): Name of the project where the run should
                go, in the form "workspace-name/project_name"
            name (Optional[str], optional): Identifier inside the project.

        Raises:
            Exception: If environment variable `NEPTUNE_TOKEN` is unset.
        """
        api_token = os.environ.get("NEPTUNE_TOKEN", None)
        if api_token is None:
            raise Exception(
                "Could not find the token for neptune logging. Make sure that the \
                            environment variable `NEPTUNE_TOKEN` is set."
            )

        self._stop_logging = bool(os.environ.get("NEPTUNE_DISABLE", 0))
        if self._stop_logging:
            return

        self.run = neptune.init_run(
            name=name,
            project=project,
            api_token=api_token,
        )

    def log(self, metrices) -> None:
        if self._stop_logging:
            return

        metrices = flatten_dict(metrices)
        metrices = jax.tree_map(to_float_if_not_string, metrices)

        for key, value in metrices.items():
            self.run[key].log(value)
