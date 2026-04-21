import inspect
from typing import Any

import hydra
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
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
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (
    RankedLogger,
    apply_wandb_multirun_metadata,
    extras,
    finalize_multiseed_run,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)
from src.utils.multirun import detect_multirun_context

log = RankedLogger(__name__, rank_zero_only=True)


def _supports_weights_only(callable_obj: object) -> bool:
    return "weights_only" in inspect.signature(callable_obj).parameters


def _cfg_value(cfg: DictConfig, runtime_cfg: DictConfig, key: str, runtime_key: str) -> Any:
    """Prefer top-level compatibility aliases when callers mutate them directly."""
    value = cfg.get(key)
    if value is not None:
        return value
    return runtime_cfg.get(runtime_key)


def _disable_pretrained_backbone_for_resume(
    cfg: DictConfig, runtime_cfg: DictConfig
) -> str | None:
    """Skip transfer-learning init when exact checkpoint resume is requested."""
    ckpt_path = _cfg_value(cfg, runtime_cfg, "ckpt_path", "ckpt_path")
    if ckpt_path in (None, ""):
        return None

    model_cfg = cfg.get("model")
    if model_cfg is None:
        return None

    pretrained_backbone_cfg = model_cfg.get("pretrained_backbone")
    if pretrained_backbone_cfg is None:
        return None

    checkpoint_path = pretrained_backbone_cfg.get("checkpoint_path")
    if checkpoint_path in (None, ""):
        return None

    with open_dict(model_cfg):
        model_cfg.pretrained_backbone.checkpoint_path = None
    return str(checkpoint_path)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    runtime_cfg = cfg.train
    multirun_context = detect_multirun_context(cfg)

    # set seed for random number generators in pytorch, numpy and python.random
    seed = _cfg_value(cfg, runtime_cfg, "seed", "seed")
    if seed is not None:
        L.seed_everything(seed, workers=True)

    skipped_pretrained_checkpoint = _disable_pretrained_backbone_for_resume(cfg, runtime_cfg)
    if skipped_pretrained_checkpoint is not None:
        log.info(
            "Skipping pretrained backbone initialization because ckpt_path resume takes "
            f"precedence: {skipped_pretrained_checkpoint}"
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    if multirun_context is not None:
        apply_wandb_multirun_metadata(cfg, group_name=multirun_context.group_name)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if runtime_cfg.get("run_train"):
        log.info("Starting training!")
        fit_kwargs = {}
        if _supports_weights_only(trainer.fit):
            fit_kwargs["weights_only"] = False
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=_cfg_value(cfg, runtime_cfg, "ckpt_path", "ckpt_path"),
            **fit_kwargs,
        )

    train_metrics = trainer.callback_metrics

    if _cfg_value(cfg, runtime_cfg, "test", "run_test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        test_kwargs = {}
        if _supports_weights_only(trainer.test):
            test_kwargs["weights_only"] = False
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path, **test_kwargs)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}
    finalize_multiseed_run(
        cfg,
        trainer,
        metric_dict,
        logger,
        context=multirun_context,
    )

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
