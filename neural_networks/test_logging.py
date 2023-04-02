import numpy as np

from .logging import NeptuneLogger


def test_neptune_logger():
    from datetime import datetime

    logger = NeptuneLogger("iss/test", name=str(datetime.now()), force_logging=True)
    logger.log({"awesome_float": 1.33})
    logger.log({"awesome_array": np.array(1.0)})
    logger.log({"awesome_string": "yay"})
