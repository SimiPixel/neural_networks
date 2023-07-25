from pathlib import Path

import numpy as np

import wandb

from .logging import NeptuneLogger, WandbLogger


def test_neptune_logger():
    from datetime import datetime

    logger = NeptuneLogger("iss/test", name=str(datetime.now()))
    logger.log({"awesome_float": 1.33})
    logger.log({"awesome_array": np.array(1.0)})
    logger.log({"awesome_string": "yay"})


def test_wandb_logger():
    wandb.init(project="TEST")

    logger = WandbLogger()
    logger.log({"awesome_float": 1.33})
    logger.log({"awesome_array": np.array(1.0)})
    logger.log({"awesome_string": "yay"})

    root = Path(__file__).parent.parent.joinpath("testing")

    logger.log_image(str(root.joinpath("image1.png")))
    logger.log_image(str(root.joinpath("image2.png")))
    logger.log_video(str(root.joinpath("video.mp4")))
