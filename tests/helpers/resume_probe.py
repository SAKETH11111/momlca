"""Checkpoint resume probe callbacks for training tests."""

from __future__ import annotations

import json
from pathlib import Path

from lightning import Callback, LightningModule, Trainer


class ResumeProbeCallback(Callback):
    """Capture the restored epoch and global step when fit starts."""

    def __init__(self) -> None:
        self.train_start_epoch: int | None = None
        self.train_start_global_step: int | None = None

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        self.train_start_epoch = trainer.current_epoch
        self.train_start_global_step = trainer.global_step


class PersistedResumeProbeCallback(ResumeProbeCallback):
    """Persist the observed resume state to disk for subprocess-based tests."""

    def __init__(self, output_path: str) -> None:
        super().__init__()
        self.output_path = Path(output_path)

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_start(trainer, pl_module)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(
                {
                    "epoch": self.train_start_epoch,
                    "global_step": self.train_start_global_step,
                }
            )
        )
