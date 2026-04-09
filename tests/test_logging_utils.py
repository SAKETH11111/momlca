"""Tests for shared logging helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import rootutils
import torch.nn as nn
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, open_dict

from src.utils.logging_utils import log_hyperparameters


@pytest.fixture
def composed_train_cfg(tmp_path: Path) -> DictConfig:
    """Minimal composed config with resolved paths for logging tests."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["logger=tensorboard"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.paths.output_dir = str(tmp_path / "hydra-out")
            cfg.paths.log_dir = str(tmp_path / "logs")

    yield cfg
    GlobalHydra.instance().clear()


def test_log_hyperparameters_includes_git_and_checkpoint_metadata(
    composed_train_cfg: DictConfig,
) -> None:
    """Hyperparameter logging should attach git + checkpoint paths for reproducibility."""
    captured: dict = {}

    def _capture(h: dict) -> None:
        captured.clear()
        captured.update(h)

    mock_logger = MagicMock()
    mock_logger.log_hyperparams.side_effect = _capture

    trainer = MagicMock()
    trainer.loggers = [mock_logger]

    model = nn.Linear(2, 1)
    object_dict = {"cfg": composed_train_cfg, "model": model, "trainer": trainer}

    log_hyperparameters(object_dict)

    assert captured["checkpoint_output_dir"].endswith("/checkpoints")
    assert (
        Path(str(captured["hydra_output_dir"]) or "")
        == Path(composed_train_cfg.paths.output_dir).resolve()
    )
    assert "git_commit" in captured
    assert len(captured["git_commit"]) == 40
    assert captured["git_commit_short"] == captured["git_commit"][:7]
    assert "task_name" in captured
    assert "model" in captured and "data" in captured


def test_log_hyperparameters_adds_git_tag_to_wandb_logger(
    composed_train_cfg: DictConfig,
) -> None:
    """W&B logger config should receive a deterministic git tag without starting a real run."""
    captured: dict = {}

    def _capture(h: dict) -> None:
        captured.clear()
        captured.update(h)

    wandb_logger = WandbLogger(
        project="moml",
        offline=True,
        save_dir=str(composed_train_cfg.paths.output_dir),
    )
    wandb_logger.log_hyperparams = MagicMock(side_effect=_capture)

    trainer = MagicMock()
    trainer.loggers = [wandb_logger]

    object_dict = {
        "cfg": composed_train_cfg,
        "model": nn.Linear(2, 1),
        "trainer": trainer,
    }

    log_hyperparameters(object_dict)

    git_tag = f"git:{captured['git_commit_short']}"
    assert captured["git_tag"] == git_tag
    assert git_tag in wandb_logger._wandb_init["tags"]


def test_log_hyperparameters_skips_when_no_loggers(composed_train_cfg: DictConfig) -> None:
    """Without loggers, logging should no-op without raising."""
    trainer = MagicMock()
    trainer.loggers = []

    object_dict = {
        "cfg": composed_train_cfg,
        "model": nn.Linear(1, 1),
        "trainer": trainer,
    }

    log_hyperparameters(object_dict)
