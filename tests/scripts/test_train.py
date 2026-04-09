"""Tests for the user-facing training launcher."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from tests.helpers.pfasbench import write_sample_pfasbench_dataset


def test_train_script_uses_canonical_config_and_persists_resolved_hydra_config(
    tmp_path: Path,
) -> None:
    """The user-facing train script should persist Hydra config and accept resume checkpoints."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / "run"
    first_probe_path = tmp_path / "resume-probe-first.json"
    second_probe_path = tmp_path / "resume-probe-second.json"

    base_command = [
        sys.executable,
        "scripts/train.py",
        "model=painn",
        "data=pfasbench",
        "train.run_test=false",
        "trainer.accelerator=cpu",
        "trainer.devices=1",
        "trainer.max_epochs=1",
        "+trainer.limit_train_batches=1",
        "+trainer.limit_val_batches=1",
        "+trainer.num_sanity_val_steps=0",
        "data.batch_size=2",
        "data.num_workers=0",
        "data.root=" + str(dataset_root),
        "data.split=random",
        "data.train_frac=0.5",
        "data.val_frac=0.25",
        "data.test_frac=0.25",
        "extras.print_config=false",
        "extras.enforce_tags=false",
        "hydra.run.dir=" + str(run_dir),
    ]

    subprocess.run(
        [
            *base_command,
            "+callbacks.resume_probe._target_=tests.helpers.resume_probe.PersistedResumeProbeCallback",
            "+callbacks.resume_probe.output_path=" + str(first_probe_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    hydra_config = run_dir / ".hydra" / "config.yaml"
    assert hydra_config.exists()

    resolved_cfg = yaml.safe_load(hydra_config.read_text())
    assert resolved_cfg["model"]["_target_"] == "gnn.models.MoMLCAModel"
    assert (
        resolved_cfg["model"]["backbone"]["_target_"] == "gnn.models.backbones.PaiNNStageBackbone"
    )
    assert resolved_cfg["data"]["_target_"] == "gnn.data.datamodules.PFASBenchDataModule"
    assert resolved_cfg["trainer"]["max_epochs"] == 1
    assert resolved_cfg["train"]["run_test"] is False

    checkpoint_dir = run_dir / "checkpoints"
    assert (checkpoint_dir / "best.ckpt").exists()
    assert (checkpoint_dir / "last.ckpt").exists()
    assert (checkpoint_dir / "epoch_000.ckpt").exists()
    first_probe = json.loads(first_probe_path.read_text())
    assert first_probe == {"epoch": 0, "global_step": 0}
    first_global_step = torch.load(
        checkpoint_dir / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )["global_step"]

    subprocess.run(
        [
            *base_command,
            "trainer.max_epochs=2",
            "ckpt_path=" + str(checkpoint_dir / "last.ckpt"),
            "+callbacks.resume_probe._target_=tests.helpers.resume_probe.PersistedResumeProbeCallback",
            "+callbacks.resume_probe.output_path=" + str(second_probe_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    assert (checkpoint_dir / "epoch_001.ckpt").exists()
    second_probe = json.loads(second_probe_path.read_text())
    assert second_probe["epoch"] == 1
    assert second_probe["global_step"] == first_global_step


@pytest.mark.skipif(
    importlib.util.find_spec("tensorboard") is None
    and importlib.util.find_spec("tensorboardX") is None,
    reason="TensorBoard runtime dependency is not installed",
)
def test_train_script_runs_with_tensorboard_logger(tmp_path: Path) -> None:
    """Canonical training should run with the TensorBoard fallback logger."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "model=painn",
            "data=pfasbench",
            "train.run_test=false",
            "trainer.accelerator=cpu",
            "trainer.devices=1",
            "trainer.max_epochs=1",
            "+trainer.limit_train_batches=1",
            "+trainer.limit_val_batches=1",
            "+trainer.num_sanity_val_steps=0",
            "data.batch_size=2",
            "data.num_workers=0",
            "data.root=" + str(dataset_root),
            "data.split=random",
            "data.train_frac=0.5",
            "data.val_frac=0.25",
            "data.test_frac=0.25",
            "logger=tensorboard",
            "extras.print_config=false",
            "extras.enforce_tags=false",
            "hydra.run.dir=" + str(run_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    hydra_config = run_dir / ".hydra" / "config.yaml"
    assert hydra_config.exists()
    resolved_cfg = yaml.safe_load(hydra_config.read_text())
    assert resolved_cfg["logger"]["tensorboard"]["save_dir"].endswith("/tensorboard/")
