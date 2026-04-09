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


def test_train_script_multirun_writes_child_and_sweep_artifacts(tmp_path: Path) -> None:
    """Canonical multirun should preserve child outputs and write sweep summaries."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    sweep_dir = tmp_path / "multirun"

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--multirun",
            "model=painn",
            "data=pfasbench",
            "seed=11,12",
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
            "data.seed=77",
            "data.train_frac=0.5",
            "data.val_frac=0.25",
            "data.test_frac=0.25",
            "extras.print_config=false",
            "extras.enforce_tags=false",
            "hydra.sweep.dir=" + str(sweep_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    child_dirs = sorted(
        path for path in sweep_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    assert [path.name for path in child_dirs] == ["0", "1"]

    child_artifacts: list[dict] = []
    for child_dir, expected_seed in zip(child_dirs, [11, 12], strict=True):
        hydra_config = child_dir / ".hydra" / "config.yaml"
        metrics_artifact = child_dir / "multiseed_metrics.json"

        assert hydra_config.exists()
        assert metrics_artifact.exists()

        resolved_cfg = yaml.safe_load(hydra_config.read_text())
        assert resolved_cfg["seed"] == expected_seed
        assert resolved_cfg["data"]["seed"] == 77

        child_metrics = json.loads(metrics_artifact.read_text())
        child_artifacts.append(child_metrics)
        assert child_metrics["seed"] == expected_seed
        assert child_metrics["data_seed"] == 77
        assert Path(child_metrics["output_dir"]) == child_dir
        assert "val/loss" in child_metrics["metrics"]

    summary_json = sweep_dir / "multiseed_summary.json"
    summary_csv = sweep_dir / "multiseed_summary.csv"
    summary_md = sweep_dir / "multiseed_summary.md"
    assert summary_json.exists()
    assert summary_csv.exists()
    assert summary_md.exists()

    summary_payload = json.loads(summary_json.read_text())
    assert summary_payload["run_count"] == 2
    assert [row["seed"] for row in summary_payload["runs"]] == [11, 12]
    assert summary_payload["aggregate_stats"]["val/loss"]["n"] == 2


def test_train_script_non_seed_multirun_does_not_emit_multiseed_artifacts(
    tmp_path: Path,
) -> None:
    """Generic Hydra sweeps should not be mislabeled as canonical multi-seed runs."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    sweep_dir = tmp_path / "generic-multirun"

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "--multirun",
            "model=painn",
            "data=pfasbench",
            "seed=11",
            "trainer.max_epochs=1,2",
            "train.run_test=false",
            "trainer.accelerator=cpu",
            "trainer.devices=1",
            "+trainer.limit_train_batches=1",
            "+trainer.limit_val_batches=1",
            "+trainer.num_sanity_val_steps=0",
            "data.batch_size=2",
            "data.num_workers=0",
            "data.root=" + str(dataset_root),
            "data.split=random",
            "data.seed=77",
            "data.train_frac=0.5",
            "data.val_frac=0.25",
            "data.test_frac=0.25",
            "extras.print_config=false",
            "extras.enforce_tags=false",
            "hydra.sweep.dir=" + str(sweep_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    child_dirs = sorted(
        path for path in sweep_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    assert [path.name for path in child_dirs] == ["0", "1"]
    assert all(not (child_dir / "multiseed_metrics.json").exists() for child_dir in child_dirs)
    assert not (sweep_dir / "multiseed_summary.json").exists()
    assert not (sweep_dir / "multiseed_summary.csv").exists()
    assert not (sweep_dir / "multiseed_summary.md").exists()
