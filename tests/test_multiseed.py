"""Tests for canonical multi-seed training helpers."""

from __future__ import annotations

import csv
import json
import math
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from src.train import train
from src.utils.logging_utils import apply_wandb_multirun_metadata, log_multiseed_summary_to_wandb
from src.utils.multirun import (
    MultirunContext,
    _expected_run_count,
    _is_seed_sweep,
    _job_num_sort_key,
    compute_summary_statistics,
    finalize_multiseed_run,
)
from tests.helpers.pfasbench import write_sample_pfasbench_dataset


def _compose_multiseed_cfg(tmp_path: Path):
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=painn", "data=pfasbench", "train=multiseed"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.paths.output_dir = str(tmp_path / "sweep" / "0")
            cfg.paths.log_dir = str(tmp_path / "logs")
            cfg.paths.data_dir = str(dataset_root)
            cfg.data.root = str(dataset_root)
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.split = "random"
            cfg.data.seed = 99
            cfg.data.train_frac = 0.5
            cfg.data.val_frac = 0.25
            cfg.data.test_frac = 0.25
            cfg.seed = 7
            cfg.train.run_test = False
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.train.multiseed.enabled = True
            cfg.train.multiseed.sweep_dir = str(tmp_path / "sweep")
            cfg.train.multiseed.job_num = "0"
            cfg.logger = None
    return cfg


def test_train_writes_multiseed_artifacts_from_src_entrypoint(tmp_path: Path) -> None:
    """Direct src.train execution should still emit the canonical multiseed artifacts."""
    cfg = _compose_multiseed_cfg(tmp_path)

    try:
        metric_dict, _ = train(cfg)
    finally:
        GlobalHydra.instance().clear()

    run_artifact = tmp_path / "sweep" / "0" / "multiseed_metrics.json"
    summary_json = tmp_path / "sweep" / "multiseed_summary.json"

    assert run_artifact.exists()
    assert summary_json.exists()
    assert "val/loss" in metric_dict

    run_payload = json.loads(run_artifact.read_text())
    summary_payload = json.loads(summary_json.read_text())

    assert run_payload["seed"] == 7
    assert run_payload["data_seed"] == 99
    assert summary_payload["run_count"] == 1
    assert summary_payload["runs"][0]["seed"] == 7


def test_compute_summary_statistics_uses_sample_std_and_ci95() -> None:
    """Aggregate statistics should use sample std and 95% CI from stored child metrics."""
    stats = compute_summary_statistics(
        [
            {"metrics": {"val/mae": 1.0, "test/mae": 4.0}},
            {"metrics": {"val/mae": 2.0, "test/mae": 5.0}},
            {"metrics": {"val/mae": 3.0, "test/mae": 6.0}},
        ]
    )

    val_mae = stats["val/mae"]
    expected_half_width = 1.96 / (3**0.5)
    assert val_mae["n"] == 3
    assert val_mae["mean"] == pytest.approx(2.0)
    assert val_mae["std"] == pytest.approx(1.0)
    assert val_mae["sem"] == pytest.approx(1.0 / (3**0.5))
    assert val_mae["ci_method"] == "normal"
    assert val_mae["ci_level"] == pytest.approx(0.95)
    assert val_mae["ci_low"] == pytest.approx(2.0 - expected_half_width)
    assert val_mae["ci_high"] == pytest.approx(2.0 + expected_half_width)
    assert val_mae["ci_half_width"] == pytest.approx(expected_half_width)
    assert val_mae["ci95"] == pytest.approx(expected_half_width)


def test_compute_summary_statistics_hides_bounds_for_low_support_metrics() -> None:
    """Low-support summaries should avoid emitting misleading confidence bounds."""
    stats = compute_summary_statistics(
        [
            {"metrics": {"val/mae": 1.0, "val/loss": 4.0}},
            {"metrics": {"val/mae": 2.0}},
        ]
    )

    val_mae = stats["val/mae"]
    assert val_mae["n"] == 2
    assert val_mae["std"] == pytest.approx(1.0 / (2**0.5))
    assert val_mae["sem"] == pytest.approx(0.5)
    assert val_mae["ci_method"] is None
    assert val_mae["ci_level"] is None
    assert val_mae["ci_low"] is None
    assert val_mae["ci_high"] is None
    assert val_mae["ci_half_width"] is None
    assert val_mae["ci95"] is None

    val_loss = stats["val/loss"]
    assert val_loss["n"] == 1
    assert val_loss["std"] is None
    assert val_loss["sem"] is None
    assert val_loss["ci_method"] is None
    assert val_loss["ci_low"] is None


def test_compute_summary_statistics_supports_deterministic_bootstrap_ci() -> None:
    """Bootstrap intervals should be available as an explicit deterministic option."""
    run_records = [
        {"metrics": {"val/mae": 1.0}},
        {"metrics": {"val/mae": 2.0}},
        {"metrics": {"val/mae": 3.0}},
        {"metrics": {"val/mae": 4.0}},
    ]
    stats_a = compute_summary_statistics(
        run_records,
        ci_method="bootstrap",
        ci_level=0.9,
        bootstrap_resamples=500,
        bootstrap_random_seed=11,
    )
    stats_b = compute_summary_statistics(
        run_records,
        ci_method="bootstrap",
        ci_level=0.9,
        bootstrap_resamples=500,
        bootstrap_random_seed=11,
    )

    metric_a = stats_a["val/mae"]
    metric_b = stats_b["val/mae"]
    assert metric_a["ci_method"] == "bootstrap"
    assert metric_a["ci_level"] == pytest.approx(0.9)
    assert metric_a["ci_low"] == pytest.approx(metric_b["ci_low"])
    assert metric_a["ci_high"] == pytest.approx(metric_b["ci_high"])
    assert metric_a["ci_half_width"] == pytest.approx(metric_b["ci_half_width"])
    assert metric_a["ci_low"] < metric_a["mean"] < metric_a["ci_high"]
    assert metric_a["ci_half_width"] == pytest.approx(
        (metric_a["ci_high"] - metric_a["ci_low"]) / 2
    )
    assert metric_a["ci95"] == pytest.approx(1.96 * metric_a["std"] / math.sqrt(metric_a["n"]))


def test_job_num_sort_key_orders_numeric_jobs_before_lexicographic_suffixes() -> None:
    """Sweep summaries should keep numeric Hydra job ids in numeric order."""
    assert sorted(["0", "1", "10", "2", "alpha"], key=_job_num_sort_key) == [
        "0",
        "1",
        "2",
        "10",
        "alpha",
    ]


def test_is_seed_sweep_uses_original_multirun_argv() -> None:
    """Detection should distinguish swept seeds from a fixed seed on another sweep axis."""
    assert _is_seed_sweep(["seed=11"], ["--multirun", "seed=11,12"])
    assert not _is_seed_sweep(
        ["seed=11", "trainer.max_epochs=1"],
        ["--multirun", "seed=11", "trainer.max_epochs=1,2"],
    )


def test_expected_run_count_tracks_multirun_cross_product() -> None:
    """Expected run count should reflect the original Hydra CLI sweep axes."""
    assert _expected_run_count(["--multirun", "seed=11,12,13"]) == 3
    assert _expected_run_count(["--multirun", "seed=11,12", "trainer.max_epochs=1,2"]) == 4
    assert _expected_run_count(["--multirun", "seed=11"]) is None


def test_finalize_multiseed_run_skips_wandb_summary_for_hydra_child_runs(
    tmp_path: Path,
) -> None:
    """Hydra child jobs should not each publish a duplicate aggregate W&B summary."""
    cfg = _compose_multiseed_cfg(tmp_path)

    experiment = types.SimpleNamespace(summary={}, log=MagicMock())
    logger = types.SimpleNamespace(_wandb_init={}, experiment=experiment)
    trainer = types.SimpleNamespace(
        checkpoint_callback=types.SimpleNamespace(best_model_path=""),
    )
    context = MultirunContext(
        output_dir=tmp_path / "sweep" / "10",
        sweep_dir=tmp_path / "sweep",
        job_num="10",
        group_name="train-multiseed-test",
        is_hydra_multirun=True,
        expected_run_count=None,
    )

    summary_payload = finalize_multiseed_run(
        cfg,
        trainer,
        {"val/loss": 1.23},
        [logger],
        context=context,
    )

    assert summary_payload is not None
    assert summary_payload["run_count"] == 1
    assert summary_payload["runs"][0]["job_num"] == "10"
    experiment.log.assert_not_called()
    assert experiment.summary == {}


def test_finalize_multiseed_run_logs_wandb_summary_once_when_hydra_sweep_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final completed Hydra seed sweep should publish one aggregate W&B summary."""
    cfg = _compose_multiseed_cfg(tmp_path)
    logger = types.SimpleNamespace(
        _wandb_init={}, experiment=types.SimpleNamespace(summary={}, log=MagicMock())
    )
    trainer = types.SimpleNamespace(checkpoint_callback=types.SimpleNamespace(best_model_path=""))

    class FakeTable:
        def __init__(self, *, columns, data) -> None:
            self.columns = columns
            self.data = data

    monkeypatch.setitem(sys.modules, "wandb", types.SimpleNamespace(Table=FakeTable))

    first_context = MultirunContext(
        output_dir=tmp_path / "sweep" / "0",
        sweep_dir=tmp_path / "sweep",
        job_num="0",
        group_name="train-multiseed-test",
        is_hydra_multirun=True,
        expected_run_count=2,
    )
    second_context = MultirunContext(
        output_dir=tmp_path / "sweep" / "1",
        sweep_dir=tmp_path / "sweep",
        job_num="1",
        group_name="train-multiseed-test",
        is_hydra_multirun=True,
        expected_run_count=2,
    )

    finalize_multiseed_run(cfg, trainer, {"val/loss": 1.23}, [logger], context=first_context)
    first_logger = logger.experiment.log.call_count
    summary_payload = finalize_multiseed_run(
        cfg,
        trainer,
        {"val/loss": 1.11},
        [logger],
        context=second_context,
    )

    assert summary_payload is not None
    assert summary_payload["is_complete"] is True
    assert summary_payload["missing_job_nums"] == []
    assert logger.experiment.log.call_count == first_logger + 1
    assert logger.experiment.summary["multiseed/run_count"] == 2


def test_finalize_multiseed_run_marks_incomplete_summaries(
    tmp_path: Path,
) -> None:
    """Missing child artifacts should be surfaced in the summary payload and file."""
    cfg = _compose_multiseed_cfg(tmp_path)
    trainer = types.SimpleNamespace(checkpoint_callback=types.SimpleNamespace(best_model_path=""))
    context = MultirunContext(
        output_dir=tmp_path / "sweep" / "0",
        sweep_dir=tmp_path / "sweep",
        job_num="0",
        group_name="train-multiseed-test",
        is_hydra_multirun=True,
        expected_run_count=3,
    )

    summary_payload = finalize_multiseed_run(cfg, trainer, {"val/loss": 1.23}, [], context=context)

    assert summary_payload is not None
    assert summary_payload["is_complete"] is False
    assert summary_payload["missing_job_nums"] == ["1", "2"]

    summary_json = tmp_path / "sweep" / "multiseed_summary.json"
    saved_payload = json.loads(summary_json.read_text())
    assert saved_payload["is_complete"] is False
    assert saved_payload["missing_job_nums"] == ["1", "2"]


def test_apply_and_log_multiseed_wandb_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline-safe W&B helpers should set grouping metadata and publish sweep summaries."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["logger=wandb"],
        )
    try:
        apply_wandb_multirun_metadata(cfg, group_name="train-multiseed-test")

        assert cfg.logger.wandb.group == "train-multiseed-test"
        assert cfg.logger.wandb.job_type == "multiseed-child"

        tables: list[object] = []

        class FakeTable:
            def __init__(self, *, columns, data) -> None:
                self.columns = columns
                self.data = data
                tables.append(self)

        fake_wandb = types.SimpleNamespace(Table=FakeTable)
        monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

        experiment = types.SimpleNamespace(summary={}, log=MagicMock())
        logger = types.SimpleNamespace(_wandb_init={}, experiment=experiment)
        payload = {
            "group_name": "train-multiseed-test",
            "run_count": 2,
            "runs": [
                {
                    "job_num": "0",
                    "seed": 11,
                    "data_seed": 77,
                    "checkpoint_path": "a",
                    "metrics": {"val/loss": 1.0},
                },
                {
                    "job_num": "1",
                    "seed": 12,
                    "data_seed": 77,
                    "checkpoint_path": "b",
                    "metrics": {"val/loss": 2.0},
                },
            ],
            "aggregate_stats": {"val/loss": {"n": 2, "mean": 1.5, "std": 0.7071, "ci95": None}},
            "artifacts": {"summary_json": "/tmp/multiseed_summary.json"},
        }

        log_multiseed_summary_to_wandb([logger], payload, prefix="multiseed")

        assert experiment.summary["multiseed/group"] == "train-multiseed-test"
        assert experiment.summary["multiseed/run_count"] == 2
        assert experiment.summary["multiseed/val/loss/mean"] == pytest.approx(1.5)
        experiment.log.assert_called_once()
        assert len(tables) == 2
        aggregate_table = next(table for table in tables if table.columns[0] == "metric")
        assert aggregate_table.columns == [
            "metric",
            "n",
            "mean",
            "std",
            "sem",
            "ci_method",
            "ci_level",
            "ci_low",
            "ci_high",
            "ci_half_width",
            "ci95",
            "ci_display",
        ]
    finally:
        GlobalHydra.instance().clear()


def test_finalize_multiseed_run_writes_ci_columns_to_csv_and_markdown(tmp_path: Path) -> None:
    """Deterministic sweep artifacts should include machine-readable and display CI fields."""
    cfg = _compose_multiseed_cfg(tmp_path)
    trainer = types.SimpleNamespace(checkpoint_callback=types.SimpleNamespace(best_model_path=""))

    contexts = [
        MultirunContext(
            output_dir=tmp_path / "sweep" / str(job_num),
            sweep_dir=tmp_path / "sweep",
            job_num=str(job_num),
            group_name="train-multiseed-test",
            is_hydra_multirun=True,
            expected_run_count=3,
        )
        for job_num in (0, 1, 2)
    ]
    for context, loss in zip(contexts, [1.0, 2.0, 3.0], strict=True):
        finalize_multiseed_run(cfg, trainer, {"val/loss": loss}, [], context=context)

    summary_csv = tmp_path / "sweep" / "multiseed_summary.csv"
    rows = list(csv.DictReader(summary_csv.read_text().splitlines()))
    assert len(rows) == 1
    row = rows[0]
    assert row["metric"] == "val/loss"
    assert row["sem"] != ""
    assert row["ci_method"] == "normal"
    assert row["ci_level"] == "0.95"
    assert row["ci_low"] != ""
    assert row["ci_high"] != ""
    assert row["ci_half_width"] != ""
    assert row["ci95"] != ""
    assert row["ci_display"] == "2.0000 +/- 1.1316"

    summary_md = tmp_path / "sweep" / "multiseed_summary.md"
    markdown = summary_md.read_text()
    assert "ci_method" in markdown
    assert "ci_display" in markdown
    assert "2.0000 +/- 1.1316" in markdown
