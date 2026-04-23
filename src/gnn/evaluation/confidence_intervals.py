"""Helpers for normalizing and rendering sweep-level confidence interval summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

INTERVAL_FIELD_NAMES: tuple[str, ...] = (
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
)


def normalize_interval_summary(
    summary: Mapping[str, Any],
) -> dict[str, int | float | str | None]:
    """Coerce one metric summary into a stable interval-field dictionary."""
    n = _coerce_int(summary.get("n"))
    mean = _coerce_float(summary.get("mean"))
    std = _coerce_float(summary.get("std"))
    sem = _coerce_float(summary.get("sem"))
    ci_method = _coerce_method(summary.get("ci_method"))
    ci_level = _coerce_float(summary.get("ci_level"))
    ci_low = _coerce_float(summary.get("ci_low"))
    ci_high = _coerce_float(summary.get("ci_high"))
    ci_half_width = _coerce_float(summary.get("ci_half_width"))
    ci95 = _coerce_float(summary.get("ci95"))

    if ci_half_width is None and ci_low is not None and ci_high is not None:
        ci_half_width = (ci_high - ci_low) / 2.0

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_method": ci_method,
        "ci_level": ci_level,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_half_width": ci_half_width,
        "ci95": ci95,
    }


def format_interval_display(summary: Mapping[str, Any], *, precision: int = 4) -> str | None:
    """Return paper-friendly `mean +/- half-width` text when interval data is available."""
    normalized = normalize_interval_summary(summary)
    mean = normalized["mean"]
    if mean is None:
        return None

    ci_half_width = normalized["ci_half_width"]
    if ci_half_width is None:
        ci_half_width = normalized["ci95"]
    if ci_half_width is None:
        return None

    return f"{mean:.{precision}f} +/- {ci_half_width:.{precision}f}"


def interval_report_fields(
    summary: Mapping[str, Any],
    *,
    prefix: str = "",
    precision: int = 4,
) -> dict[str, int | float | str | None]:
    """Return interval fields plus `ci_display`, optionally prefixed for table columns."""
    normalized = normalize_interval_summary(summary)
    fields: dict[str, int | float | str | None] = dict(normalized)
    fields["ci_display"] = format_interval_display(summary, precision=precision)

    if prefix == "":
        return fields
    return {f"{prefix}{field_name}": value for field_name, value in fields.items()}


def flatten_confidence_interval_metadata(
    interval_map: Mapping[str, Mapping[str, Any]],
    *,
    precision: int = 4,
) -> dict[str, int | float | str | None]:
    """Flatten per-metric interval mappings into scalar table columns."""
    flattened: dict[str, int | float | str | None] = {}
    for metric_name in sorted(interval_map):
        metric_summary = interval_map[metric_name]
        if not isinstance(metric_summary, Mapping):
            continue
        flattened.update(
            interval_report_fields(
                metric_summary,
                prefix=f"{metric_name}_",
                precision=precision,
            )
        )
    return flattened


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(as_float):
        return None
    as_int = int(as_float)
    if float(as_int) != as_float:
        return None
    return as_int


def _coerce_method(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if stripped == "":
        return None
    return stripped


__all__ = [
    "INTERVAL_FIELD_NAMES",
    "flatten_confidence_interval_metadata",
    "format_interval_display",
    "interval_report_fields",
    "normalize_interval_summary",
]
