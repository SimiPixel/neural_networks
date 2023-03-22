from abc import ABC, abstractmethod
from typing import Optional

import jax
import neptune.new as neptune
import optax
from tree_utils import PyTree

# An arbitrarily nested dictionary with jax.Array leaves; Or strings
NestedDict = PyTree


class Logger(ABC):
    @abstractmethod
    def log(self, metrics: NestedDict):
        pass

    def close(self):
        pass


api_token = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfd\
    XJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MGJlN2E2Mi0yYzh\
        kLTRmYmEtOWJiNC01ZTViYTFkZmQ0YzAifQ=="


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
    def __init__(self, project: str, name: Optional[str] = None):
        self.run = neptune.init_run(
            name=name,
            project=project,
            api_token=api_token,
        )

    def log(self, metrices):
        metrices = flatten_dict(metrices)
        metrices = jax.tree_map(to_float_if_not_string, metrices)

        for key, value in metrices.items():
            self.run[key].log(value)
