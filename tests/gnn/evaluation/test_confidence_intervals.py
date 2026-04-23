"""Tests for confidence interval normalization and display helpers."""

from __future__ import annotations

from gnn.evaluation.confidence_intervals import format_interval_display, normalize_interval_summary


def test_format_interval_display_marks_ci95_fallback_explicitly() -> None:
    display = format_interval_display({"mean": 1.5, "ci95": 0.2})
    assert display == "1.5000 +/- 0.2000 (ci95 fallback)"


def test_format_interval_display_prefers_interval_half_width_when_available() -> None:
    display = format_interval_display({"mean": 1.5, "ci_half_width": 0.2, "ci95": 0.4})
    assert display == "1.5000 +/- 0.2000"


def test_normalize_interval_summary_rejects_invalid_counts_and_negative_scales() -> None:
    normalized = normalize_interval_summary(
        {
            "n": True,
            "std": -0.3,
            "sem": -0.1,
            "ci_level": -0.95,
            "ci_half_width": -0.02,
            "ci95": -0.02,
        }
    )
    assert normalized["n"] is None
    assert normalized["std"] is None
    assert normalized["sem"] is None
    assert normalized["ci_level"] is None
    assert normalized["ci_half_width"] is None
    assert normalized["ci95"] is None


def test_normalize_interval_summary_rejects_negative_integer_counts() -> None:
    normalized = normalize_interval_summary({"n": -1, "mean": 2.0})
    assert normalized["n"] is None
    assert normalized["mean"] == 2.0
