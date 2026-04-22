import inspect
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import rootutils
from lightning import LightningDataModule, LightningModule, Trainer
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

from gnn.evaluation.export import (
    build_prediction_records,
    export_prediction_records,
    maybe_log_prediction_artifact,
)
from src.utils import (
    RankedLogger,
    extras,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _supports_kwarg(callable_obj: object, parameter_name: str) -> bool:
    return parameter_name in inspect.signature(callable_obj).parameters


def _supports_weights_only(callable_obj: object) -> bool:
    return _supports_kwarg(callable_obj, "weights_only")


def _supports_prediction_collection(trainer: Trainer) -> bool:
    strategy = getattr(trainer, "strategy", None)
    if strategy is None:
        return True

    strategy_name = str(getattr(strategy, "strategy_name", "")).lower()
    launcher = getattr(strategy, "launcher", None)
    launcher_name = launcher.__class__.__name__.lower() if launcher is not None else ""
    disallowed_tokens = ("spawn", "fork")

    return not (
        any(token in strategy_name for token in disallowed_tokens)
        or any(token in launcher_name for token in disallowed_tokens)
    )


def _strategy_name(trainer: Trainer) -> str:
    return str(getattr(getattr(trainer, "strategy", None), "strategy_name", "unknown"))


def _predict_kwargs_for_export(trainer: Trainer) -> dict[str, Any]:
    predict_kwargs: dict[str, Any] = {}
    if _supports_weights_only(trainer.predict):
        # Keep Lightning checkpoint loading behavior compatible with train/test paths.
        predict_kwargs["weights_only"] = False

    if _supports_kwarg(trainer.predict, "return_predictions"):
        if not _supports_prediction_collection(trainer):
            raise RuntimeError(
                "Prediction export requires return_predictions=True, but strategy "
                f"'{_strategy_name(trainer)}' does not support prediction collection. "
                "Use a non-spawn/fork strategy or disable export_predictions."
            )
        predict_kwargs["return_predictions"] = True
    return predict_kwargs


def _disable_pretrained_backbone_for_resume(cfg: DictConfig) -> str | None:
    """Skip transfer-learning backbone init when exact checkpoint eval is requested."""
    ckpt_path = cfg.get("ckpt_path")
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


def _to_scalar(value: Any) -> float | None:
    if hasattr(value, "detach"):
        detached = value.detach()
        if detached.numel() != 1:
            return None
        return float(detached.cpu().item())
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _summarize_test_metrics(metric_dict: Mapping[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric_name, metric_value in metric_dict.items():
        if not str(metric_name).startswith("test/"):
            continue
        scalar = _to_scalar(metric_value)
        if scalar is None:
            continue
        summary[str(metric_name)] = scalar
    return summary


def _flatten_prediction_outputs(predictions: Any) -> list[dict[str, Any]]:
    if predictions is None:
        return []

    flattened: list[dict[str, Any]] = []
    queue: deque[Any] = deque([predictions])
    while queue:
        current = queue.popleft()
        if isinstance(current, Mapping):
            flattened.append(dict(current))
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
            queue.extend(list(current))
            continue
        raise TypeError(
            "Prediction export expects mapping outputs from model.predict_step(), "
            f"received unsupported type: {type(current).__name__}"
        )
    return flattened


@task_wrapper
def evaluate(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Tuple[dict, dict] with metrics and dict with all instantiated objects.
    """
    assert cfg.ckpt_path

    skipped_pretrained_checkpoint = _disable_pretrained_backbone_for_resume(cfg)
    if skipped_pretrained_checkpoint is not None:
        log.info(
            "Skipping pretrained backbone initialization because ckpt_path evaluation takes "
            f"precedence: {skipped_pretrained_checkpoint}"
        )

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting testing!")
    test_kwargs: dict[str, Any] = {}
    if _supports_weights_only(trainer.test):
        test_kwargs["weights_only"] = False
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path, **test_kwargs)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics

    metric_summary = _summarize_test_metrics(metric_dict)
    if metric_summary:
        summary_text = ", ".join(
            f"{metric_name}={metric_value:.6f}"
            for metric_name, metric_value in sorted(metric_summary.items())
        )
        log.info(f"Evaluation metric summary: {summary_text}")

    if cfg.get("export_predictions", False):
        split_name = str(cfg.get("prediction_split", "test"))
        if split_name != "test":
            raise ValueError(
                "Evaluation prediction export currently supports only prediction_split='test'"
            )

        predict_kwargs = _predict_kwargs_for_export(trainer)

        log.info("Collecting test predictions for export...")
        raw_predictions = trainer.predict(
            model=model,
            datamodule=datamodule,
            ckpt_path=cfg.ckpt_path,
            **predict_kwargs,
        )
        if raw_predictions is None:
            raise RuntimeError(
                "Prediction export expected collected prediction outputs but received None. "
                "This usually indicates return_predictions was disabled by the current "
                "strategy/runtime. Disable export_predictions or use a supported strategy."
            )
        prediction_batches = _flatten_prediction_outputs(raw_predictions)

        property_names = getattr(model, "property_names", None)
        records = build_prediction_records(
            prediction_batches=prediction_batches,
            split_name=split_name,
            checkpoint_path=str(cfg.ckpt_path),
            property_names=list(property_names) if property_names is not None else None,
        )

        export_dir = Path(str(cfg.get("prediction_export_dir", cfg.paths.output_dir)))
        export_path = export_prediction_records(
            records=records,
            output_dir=export_dir,
            split_name=split_name,
            checkpoint_path=str(cfg.ckpt_path),
            property_names=list(property_names) if property_names is not None else None,
        )
        log.info(f"Exported {len(records)} {split_name} predictions to {export_path}")

        if cfg.get("log_prediction_artifact", False):
            maybe_log_prediction_artifact(
                loggers=logger,
                prediction_path=export_path,
                records=records,
                artifact_name=str(cfg.get("prediction_artifact_name", "eval-predictions")),
                split_name=split_name,
                max_rows=int(cfg.get("prediction_table_rows", 25)),
            )

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    main()
