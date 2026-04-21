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
from tests.helpers.pretrained_artifacts import (
    TRACKED_PAINN_STAGE_ARTIFACT_RELATIVE_PATH,
    require_tracked_painn_stage_artifact,
)


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
        "model=gin",
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
    assert resolved_cfg["model"]["backbone"]["_target_"] == "gnn.models.backbones.GINBackbone"
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


def test_train_script_supports_painn_fast_dev_run(tmp_path: Path) -> None:
    """The train script should run a fast-dev PaiNN smoke run."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / "painn-fast-dev-run"

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
    assert resolved_cfg["model"]["backbone"]["_target_"] == "gnn.models.backbones.PaiNNBackbone"
    assert (run_dir / "checkpoints" / "last.ckpt").exists()


def test_train_script_accepts_finetune_config_without_overloading_resume_ckpt(
    tmp_path: Path,
) -> None:
    """The canonical script should accept pretrained-backbone init separately from ckpt_path."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / "finetune-run"
    pretrained_path = tmp_path / "pretrained-backbone.pt"
    torch.save(
        {
            "backbone.linear.weight": torch.full((4, 22), 1.75),
            "backbone.linear.bias": torch.full((4,), -0.25),
        },
        pretrained_path,
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "model=momlca",
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
            "+model.backbone._target_=tests.helpers.transfer_learning.TinyBackbone",
            "+model.backbone.input_dim=22",
            "+model.backbone.hidden_dim=4",
            "model.learning_rate=0.0001",
            "model.pretrained_backbone.checkpoint_path=" + str(pretrained_path),
            "model.pretrained_backbone.checkpoint_format=state_dict",
            "model.pretrained_backbone.backbone_key_prefix=backbone.",
            "model.pretrained_backbone.freeze_backbone=true",
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
    assert resolved_cfg["ckpt_path"] == "${train.ckpt_path}"
    assert resolved_cfg["train"]["ckpt_path"] is None
    assert resolved_cfg["model"]["learning_rate"] == 0.0001
    assert resolved_cfg["model"]["pretrained_backbone"]["checkpoint_path"] == str(pretrained_path)
    assert resolved_cfg["model"]["pretrained_backbone"]["freeze_backbone"] is True

    checkpoint_dir = run_dir / "checkpoints"
    last_checkpoint = torch.load(
        checkpoint_dir / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert torch.equal(
        last_checkpoint["state_dict"]["backbone.linear.weight"],
        torch.full((4, 22), 1.75),
    )
    assert torch.equal(
        last_checkpoint["state_dict"]["backbone.linear.bias"],
        torch.full((4,), -0.25),
    )


def test_train_script_resolves_relative_pretrained_checkpoint_paths(
    tmp_path: Path,
) -> None:
    """Fine-tune presets should accept repo-relative checkpoint paths under Hydra run dirs."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / "relative-finetune-run"
    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    pretrained_path = pretrained_dir / "pretrained-backbone.pt"
    torch.save(
        {
            "backbone.node_projection.weight": torch.full((128, 22), 2.25),
            "backbone.node_projection.bias": torch.full((128,), 0.5),
        },
        pretrained_path,
    )
    relative_pretrained_path = pretrained_path.relative_to(tmp_path)
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "train.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "experiment=pfasbench_finetune_momlca",
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
            "model.pretrained_backbone.checkpoint_path=" + relative_pretrained_path.as_posix(),
            "model.pretrained_backbone.checkpoint_format=state_dict",
            "model.pretrained_backbone.freeze_backbone=true",
            "extras.print_config=false",
            "extras.enforce_tags=false",
            "hydra.run.dir=" + str(run_dir),
        ],
        check=True,
        cwd=tmp_path,
    )

    hydra_config = run_dir / ".hydra" / "config.yaml"
    assert hydra_config.exists()

    resolved_cfg = yaml.safe_load(hydra_config.read_text())
    assert resolved_cfg["model"]["_target_"] == "gnn.models.MoMLCAModel"
    assert resolved_cfg["model"]["learning_rate"] == 0.0001
    assert (
        resolved_cfg["model"]["pretrained_backbone"]["checkpoint_path"]
        == relative_pretrained_path.as_posix()
    )
    assert resolved_cfg["model"]["backbone"]["use_positions"] is False

    last_checkpoint = torch.load(
        run_dir / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert torch.equal(
        last_checkpoint["state_dict"]["backbone.node_projection.weight"],
        torch.full((128, 22), 2.25),
    )
    assert torch.equal(
        last_checkpoint["state_dict"]["backbone.node_projection.bias"],
        torch.full((128,), 0.5),
    )


@pytest.mark.parametrize(
    ("experiment_name", "run_dir_name"),
    [
        ("pfasbench_finetune", "tracked-finetune-run"),
        ("pfasbench_finetune_momlca", "tracked-finetune-momlca-run"),
    ],
)
def test_train_script_uses_tracked_real_pretrained_artifact_from_finetune_preset(
    tmp_path: Path,
    experiment_name: str,
    run_dir_name: str,
) -> None:
    """Both fine-tune presets should load the tracked DVC artifact from a Hydra run dir."""
    artifact_path = require_tracked_painn_stage_artifact()
    artifact_state = torch.load(artifact_path, map_location="cpu", weights_only=True)
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / run_dir_name
    project_root = Path(__file__).resolve().parents[2]
    script_path = project_root / "scripts" / "train.py"

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            f"experiment={experiment_name}",
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
            "model.pretrained_backbone.freeze_backbone=true",
            "extras.print_config=false",
            "extras.enforce_tags=false",
            "hydra.run.dir=" + str(run_dir),
        ],
        check=True,
        cwd=project_root,
    )

    hydra_config = run_dir / ".hydra" / "config.yaml"
    assert hydra_config.exists()

    resolved_cfg = yaml.safe_load(hydra_config.read_text())
    assert (
        resolved_cfg["model"]["pretrained_backbone"]["checkpoint_path"]
        == TRACKED_PAINN_STAGE_ARTIFACT_RELATIVE_PATH
    )
    assert resolved_cfg["model"]["pretrained_backbone"]["checkpoint_format"] == "state_dict"
    assert resolved_cfg["model"]["backbone"]["use_positions"] is False
    assert resolved_cfg["model"]["pretrained_backbone"]["freeze_backbone"] is True

    last_checkpoint = torch.load(
        run_dir / "checkpoints" / "last.ckpt",
        map_location="cpu",
        weights_only=False,
    )
    assert torch.equal(
        last_checkpoint["state_dict"]["backbone.node_projection.weight"],
        artifact_state["backbone.node_projection.weight"],
    )
    assert torch.equal(
        last_checkpoint["state_dict"]["backbone.node_projection.bias"],
        artifact_state["backbone.node_projection.bias"],
    )


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
