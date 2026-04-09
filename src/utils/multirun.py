"""Utilities for canonical Hydra multi-seed training sweeps."""

from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from lightning import Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.utils import pylogger
from src.utils.logging_utils import log_multiseed_summary_to_wandb

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


@dataclass(frozen=True)
class MultirunContext:
    """Resolved runtime metadata for a canonical multi-seed sweep."""

    output_dir: Path
    sweep_dir: Path
    job_num: str
    group_name: str
    is_hydra_multirun: bool
    expected_run_count: int | None = None


def detect_multirun_context(cfg: DictConfig) -> MultirunContext | None:
    """Return multi-run context when the current execution should emit sweep artifacts."""
    output_dir_value = OmegaConf.select(cfg, "paths.output_dir", default=None)
    if not output_dir_value:
        return None

    output_dir = Path(str(output_dir_value))
    multiseed_enabled = OmegaConf.select(cfg, "train.multiseed.enabled", default=None)

    hydra_mode = _hydra_mode()
    if hydra_mode == RunMode.MULTIRUN:
        hydra_cfg = HydraConfig.get()
        task_overrides = tuple(
            str(item) for item in OmegaConf.select(hydra_cfg, "overrides.task", default=[])
        )
        if not _is_seed_sweep(task_overrides, sys.argv[1:]):
            return None
        sweep_dir_value = OmegaConf.select(hydra_cfg, "sweep.dir", default=None)
        job_num_value = OmegaConf.select(hydra_cfg, "job.num", default=output_dir.name)
        is_hydra_multirun = True
        expected_run_count = _expected_run_count(sys.argv[1:])
    elif multiseed_enabled is True:
        sweep_dir_value = OmegaConf.select(cfg, "train.multiseed.sweep_dir", default=None)
        job_num_value = OmegaConf.select(cfg, "train.multiseed.job_num", default=output_dir.name)
        is_hydra_multirun = False
        expected_run_count = None
    else:
        return None

    sweep_dir = Path(str(sweep_dir_value)) if sweep_dir_value else output_dir.parent
    job_num = str(job_num_value)

    task_name = str(OmegaConf.select(cfg, "task_name", default="train"))
    return MultirunContext(
        output_dir=output_dir,
        sweep_dir=sweep_dir,
        job_num=job_num,
        group_name=_build_group_name(task_name, sweep_dir),
        is_hydra_multirun=is_hydra_multirun,
        expected_run_count=expected_run_count,
    )


def finalize_multiseed_run(
    cfg: DictConfig,
    trainer: Trainer,
    metric_dict: Mapping[str, Any],
    loggers: Sequence[Logger],
    *,
    context: MultirunContext | None = None,
) -> dict[str, Any] | None:
    """Persist per-run metrics and refresh sweep-level aggregate artifacts."""
    multirun_context = context or detect_multirun_context(cfg)
    if multirun_context is None:
        return None

    run_artifact_path = _write_run_metrics_artifact(cfg, trainer, metric_dict, multirun_context)
    run_records = _load_run_records(multirun_context, cfg)
    aggregate_stats = compute_summary_statistics(run_records)
    missing_job_nums = _missing_job_nums(run_records, multirun_context.expected_run_count)
    is_complete = not missing_job_nums
    artifact_paths = _write_summary_artifacts(
        cfg,
        multirun_context,
        run_records,
        aggregate_stats,
        expected_run_count=multirun_context.expected_run_count,
        is_complete=is_complete,
        missing_job_nums=missing_job_nums,
    )

    summary_payload = {
        "group_name": multirun_context.group_name,
        "run_count": len(run_records),
        "expected_run_count": multirun_context.expected_run_count,
        "is_complete": is_complete,
        "missing_job_nums": missing_job_nums,
        "runs": run_records,
        "aggregate_stats": aggregate_stats,
        "artifacts": {
            "run_metrics": str(run_artifact_path),
            **{name: str(path) for name, path in artifact_paths.items()},
        },
    }
    prefix = str(OmegaConf.select(cfg, "train.multiseed.wandb_summary_prefix", default="multiseed"))
    if missing_job_nums:
        log.warning(
            "Multi-seed summary incomplete for "
            f"{multirun_context.group_name}; missing child runs: {', '.join(missing_job_nums)}"
        )

    if not multirun_context.is_hydra_multirun or (
        multirun_context.expected_run_count is not None
        and is_complete
        and _claim_wandb_summary_logging(multirun_context)
    ):
        log_multiseed_summary_to_wandb(loggers, summary_payload, prefix=prefix)
    return summary_payload


def compute_summary_statistics(
    run_records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    """Compute aggregate statistics for every scalar metric emitted by child runs."""
    metric_names = sorted(
        {metric_name for record in run_records for metric_name in (record.get("metrics") or {})}
    )
    summaries: dict[str, dict[str, float | int | None]] = {}
    for metric_name in metric_names:
        values = [
            float(value)
            for record in run_records
            for value in [(record.get("metrics") or {}).get(metric_name)]
            if value is not None
        ]
        if not values:
            continue

        count = len(values)
        mean = float(statistics.fmean(values))
        std = float(statistics.stdev(values)) if count >= 2 else None
        ci95 = None
        if count >= 3 and std is not None:
            ci95 = float(1.96 * std / math.sqrt(count))

        summaries[metric_name] = {
            "n": count,
            "mean": mean,
            "std": std,
            "ci95": ci95,
        }
    return summaries


def _hydra_mode() -> RunMode | None:
    if not HydraConfig.initialized():
        return None
    try:
        return HydraConfig.get().mode
    except Exception:
        return None


def _build_group_name(task_name: str, sweep_dir: Path) -> str:
    safe_task = task_name.replace("/", "-")
    return f"{safe_task}-multiseed-{sweep_dir.name}"


def _is_seed_sweep(task_overrides: Sequence[str], argv: Sequence[str]) -> bool:
    if not task_overrides:
        return False

    seed_keys = {"seed", "train.seed"}
    current_keys = {override.partition("=")[0].lstrip("+") for override in task_overrides}
    if not current_keys & seed_keys:
        return False

    for arg in argv:
        key, separator, value = arg.partition("=")
        if separator and key.lstrip("+") in seed_keys and _looks_like_sweep_value(value):
            return True

    return current_keys <= seed_keys


def _looks_like_sweep_value(value: str) -> bool:
    return "," in value or "(" in value


def _expected_run_count(argv: Sequence[str]) -> int | None:
    sweep_sizes: list[int] = []
    for arg in argv:
        key, separator, value = arg.partition("=")
        if not separator or key.startswith("hydra."):
            continue
        sweep_size = _sweep_value_size(value)
        if sweep_size is not None and sweep_size > 1:
            sweep_sizes.append(sweep_size)

    if not sweep_sizes:
        return None

    total = 1
    for sweep_size in sweep_sizes:
        total *= sweep_size
    return total


def _sweep_value_size(value: str) -> int | None:
    if value.startswith("choice(") and value.endswith(")"):
        return len(_split_top_level_commas(value[7:-1]))
    if value.startswith("range(") and value.endswith(")"):
        parts = _split_top_level_commas(value[6:-1])
        if len(parts) not in {2, 3}:
            return None
        try:
            start = float(parts[0])
            stop = float(parts[1])
            step = float(parts[2]) if len(parts) == 3 else 1.0
        except ValueError:
            return None
        if step == 0:
            return None
        span = (stop - start) / step
        if span <= 0:
            return 0
        return int(math.ceil(span))
    comma_parts = _split_top_level_commas(value)
    return len(comma_parts) if len(comma_parts) > 1 else 1


def _split_top_level_commas(value: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in value:
        if char == "," and depth == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        current.append(char)
    parts.append("".join(current).strip())
    return [part for part in parts if part]


def _write_run_metrics_artifact(
    cfg: DictConfig,
    trainer: Trainer,
    metric_dict: Mapping[str, Any],
    context: MultirunContext,
) -> Path:
    artifact_path = context.output_dir / _artifact_name(
        cfg, "metrics_filename", "multiseed_metrics.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    data_seed = OmegaConf.select(cfg, "data.seed", default=None)
    train_seed = OmegaConf.select(cfg, "seed", default=None)
    if train_seed is None:
        train_seed = OmegaConf.select(cfg, "train.seed", default=None)

    checkpoint_callback = getattr(trainer, "checkpoint_callback", None)
    best_checkpoint = getattr(checkpoint_callback, "best_model_path", None)
    best_checkpoint_path = str(best_checkpoint) if best_checkpoint else None
    last_checkpoint_path = context.output_dir / "checkpoints" / "last.ckpt"

    payload = {
        "seed": train_seed,
        "data_seed": data_seed,
        "job_num": context.job_num,
        "group_name": context.group_name,
        "output_dir": str(context.output_dir),
        "sweep_dir": str(context.sweep_dir),
        "checkpoint_path": best_checkpoint_path,
        "last_checkpoint_path": (
            str(last_checkpoint_path) if last_checkpoint_path.exists() else None
        ),
        "metrics": _scalar_metrics(metric_dict),
    }
    artifact_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    log.info("Wrote multi-seed child metrics to %s", artifact_path)
    return artifact_path


def _load_run_records(context: MultirunContext, cfg: DictConfig) -> list[dict[str, Any]]:
    artifact_name = _artifact_name(cfg, "metrics_filename", "multiseed_metrics.json")
    artifact_paths = sorted(
        context.sweep_dir.glob(f"*/{artifact_name}"),
        key=lambda path: _job_num_sort_key(path.parent.name),
    )
    records: list[dict[str, Any]] = []
    for artifact_path in artifact_paths:
        try:
            records.append(json.loads(artifact_path.read_text()))
        except json.JSONDecodeError:
            log.warning("Skipping unreadable multi-seed artifact: %s", artifact_path)
    records.sort(key=lambda record: _job_num_sort_key(record.get("job_num", "")))
    return records


def _job_num_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    if text.isdigit():
        return (0, int(text))
    return (1, text)


def _missing_job_nums(
    run_records: Sequence[Mapping[str, Any]],
    expected_run_count: int | None,
) -> list[str]:
    if expected_run_count is None:
        return []

    observed = {
        int(str(record.get("job_num")))
        for record in run_records
        if str(record.get("job_num", "")).isdigit()
    }
    return [str(job_num) for job_num in range(expected_run_count) if job_num not in observed]


def _claim_wandb_summary_logging(context: MultirunContext) -> bool:
    marker_path = context.sweep_dir / ".multiseed_wandb_summary_logged"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker_path.open("x", encoding="utf-8") as handle:
            handle.write(f"{context.job_num}\n")
    except FileExistsError:
        return False
    return True


def _write_summary_artifacts(
    cfg: DictConfig,
    context: MultirunContext,
    run_records: Sequence[Mapping[str, Any]],
    aggregate_stats: Mapping[str, Mapping[str, float | int | None]],
    *,
    expected_run_count: int | None,
    is_complete: bool,
    missing_job_nums: Sequence[str],
) -> dict[str, Path]:
    context.sweep_dir.mkdir(parents=True, exist_ok=True)

    summary_json = context.sweep_dir / _artifact_name(cfg, "summary_json", "multiseed_summary.json")
    summary_csv = context.sweep_dir / _artifact_name(cfg, "summary_csv", "multiseed_summary.csv")
    summary_md = context.sweep_dir / _artifact_name(cfg, "summary_md", "multiseed_summary.md")

    summary_payload = {
        "group_name": context.group_name,
        "run_count": len(run_records),
        "expected_run_count": expected_run_count,
        "is_complete": is_complete,
        "missing_job_nums": list(missing_job_nums),
        "runs": list(run_records),
        "aggregate_stats": dict(aggregate_stats),
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n")
    _write_summary_csv(summary_csv, aggregate_stats)
    _write_summary_markdown(summary_md, run_records, aggregate_stats)

    log.info("Updated multi-seed summary artifacts in %s", context.sweep_dir)
    return {
        "summary_json": summary_json,
        "summary_csv": summary_csv,
        "summary_md": summary_md,
    }


def _write_summary_csv(
    path: Path,
    aggregate_stats: Mapping[str, Mapping[str, float | int | None]],
) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "n", "mean", "std", "ci95"])
        writer.writeheader()
        for metric_name, stats in aggregate_stats.items():
            writer.writerow({"metric": metric_name, **stats})


def _write_summary_markdown(
    path: Path,
    run_records: Sequence[Mapping[str, Any]],
    aggregate_stats: Mapping[str, Mapping[str, float | int | None]],
) -> None:
    run_rows = _run_table_rows(run_records)
    aggregate_rows = [
        [
            metric_name,
            _format_markdown_value(stats.get("n")),
            _format_markdown_value(stats.get("mean")),
            _format_markdown_value(stats.get("std")),
            _format_markdown_value(stats.get("ci95")),
        ]
        for metric_name, stats in aggregate_stats.items()
    ]
    lines = [
        "# Multi-Seed Training Summary",
        "",
        "## Aggregate Metrics",
        "",
        _markdown_table(
            ["metric", "n", "mean", "std", "ci95"],
            aggregate_rows,
        ),
        "",
        "## Per-Run Metrics",
        "",
        _markdown_table(
            ["job_num", "seed", "data_seed", "checkpoint_path", "metrics"],
            run_rows,
        ),
    ]
    path.write_text("\n".join(lines).rstrip() + "\n")


def _run_table_rows(run_records: Sequence[Mapping[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for record in run_records:
        metrics = record.get("metrics") or {}
        metric_summary = ", ".join(
            f"{metric}={_format_markdown_value(value)}" for metric, value in sorted(metrics.items())
        )
        rows.append(
            [
                str(record.get("job_num", "")),
                _format_markdown_value(record.get("seed")),
                _format_markdown_value(record.get("data_seed")),
                str(record.get("checkpoint_path") or ""),
                metric_summary,
            ]
        )
    return rows


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = ["| " + " | ".join(row) + " |" for row in rows]
    if not body_rows:
        body_rows = ["| " + " | ".join([""] * len(headers)) + " |"]
    return "\n".join([header_row, separator_row, *body_rows])


def _artifact_name(cfg: DictConfig, key: str, default: str) -> str:
    return str(OmegaConf.select(cfg, f"train.multiseed.{key}", default=default))


def _scalar_metrics(metric_dict: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name, metric_value in metric_dict.items():
        scalar = _to_scalar(metric_value)
        if scalar is not None:
            metrics[str(metric_name)] = scalar
    return metrics


def _to_scalar(value: Any) -> float | None:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
