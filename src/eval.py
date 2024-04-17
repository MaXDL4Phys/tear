from typing import List, Tuple
import hydra
# from clip import clip
import clip


from src import utils
from src.models.components.clip_modules import ImageCLIP, TextCLIP, FrameAggregation
import pyrootutils
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger as LightningLoggerBase
from omegaconf import DictConfig
# from lightning.pytorch.accelerators import find_usable_cuda_devices
import os
import torch


pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/pyrootutils
# ------------------------------------------------------------------------------------ #



log = utils.get_pylogger(__name__)


@utils.task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """
    :param cfg: A dictionary-like object that contains the configuration settings for the evaluation.
    :return: A tuple consisting of a dictionary containing the evaluation metrics and a dictionary containing the objects used in the evaluation.

    """

    if not cfg.ckpt_path:
        log.warning("No checkpoint provided!")


    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # clip
    clip_model, clip_state_dict = clip.load(
        cfg.model.network.arch,
        device="cuda",
        jit=False,
    )
    clip_model = clip_model.float()

    image_model = ImageCLIP(clip_model)
    text_model = TextCLIP(clip_model)
    frame_aggregation = FrameAggregation(
        method=cfg.model.network.frame_aggregation_method
    )

    models = {
        "clip": clip_model,
        "image": image_model,
        "text": text_model,
        "frame_aggregation": frame_aggregation,
    }

    extra_args = {
        "dataset": cfg.data.dataset,
        "n_frames": cfg.data.n_frames,
        "train_file": cfg.data.train_file,
        "num_classes": datamodule.num_classes,
        "output_dir": cfg.paths.output_dir,
        "limit_classes": cfg.data.limit_classes,
        # "task": cfg.experiment.task,
    }

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, model=models, extra_args=extra_args)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    log.info("Testing finished!")
    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """
    The `main` method is the entry point of the program. It takes a `cfg` parameter of type `DictConfig` and does not return any value.

    :param cfg: A dictionary-like object containing the configuration settings for the program.
    :return: None
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)


    utils.extras(cfg)
    evaluate(cfg)


if __name__ == "__main__":

    main()