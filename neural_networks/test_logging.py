import numpy as np

from .logging import NeptuneLogger


def test_neptune_logger():
    logger = NeptuneLogger("iss/test")
    logger.log({"awesome_float": 1.33})
    logger.log({"awesome_array": np.array(1.0)})
    logger.log({"awesome_string": "yay"})
