"""Paired significance helpers for aligned regression prediction comparisons."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

PairedTestName = Literal["wilcoxon", "ttest_rel"]


@dataclass(frozen=True)
class PairedSignificanceResult:
    """Result container for one paired significance test."""

    sample_count: int
    test_name: str
    statistic: float
    p_value: float


def run_paired_significance_test(
    first_errors: np.ndarray,
    second_errors: np.ndarray,
    *,
    test_name: PairedTestName = "wilcoxon",
) -> PairedSignificanceResult:
    """Run a paired significance test on aligned per-example error arrays."""
    if first_errors.shape != second_errors.shape:
        raise ValueError(
            "Paired significance requires equal-length error arrays "
            f"({first_errors.shape} != {second_errors.shape})"
        )

    mask = np.isfinite(first_errors) & np.isfinite(second_errors)
    first = np.asarray(first_errors[mask], dtype=float)
    second = np.asarray(second_errors[mask], dtype=float)
    sample_count = int(first.shape[0])
    if sample_count == 0:
        raise ValueError("Paired significance requires at least one aligned finite sample")

    if sample_count < 2:
        return PairedSignificanceResult(
            sample_count=sample_count,
            test_name=f"{test_name}(insufficient_samples)",
            statistic=float("nan"),
            p_value=float("nan"),
        )

    if test_name == "wilcoxon":
        # scipy.stats.wilcoxon raises when every paired difference is zero.
        if np.all(first - second == 0.0):
            return PairedSignificanceResult(
                sample_count=sample_count,
                test_name="wilcoxon",
                statistic=0.0,
                p_value=1.0,
            )
        result = wilcoxon(first, second, alternative="two-sided", zero_method="wilcox")
        return PairedSignificanceResult(
            sample_count=sample_count,
            test_name="wilcoxon",
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
        )

    if test_name == "ttest_rel":
        # scipy.stats.ttest_rel returns nan for equal paired arrays because the
        # denominator variance is zero; report a deterministic non-significant result.
        if np.all(first - second == 0.0):
            return PairedSignificanceResult(
                sample_count=sample_count,
                test_name="ttest_rel",
                statistic=0.0,
                p_value=1.0,
            )
        result = ttest_rel(first, second, nan_policy="omit")
        return PairedSignificanceResult(
            sample_count=sample_count,
            test_name="ttest_rel",
            statistic=float(result.statistic),
            p_value=float(result.pvalue),
        )

    raise ValueError(f"Unsupported test_name={test_name!r}; expected 'wilcoxon' or 'ttest_rel'")


def build_pairwise_significance_table(
    *,
    y_true: np.ndarray,
    predictions_by_model: dict[str, np.ndarray],
    property_names: list[str],
    metric_proxy: str = "absolute_error",
    test_name: PairedTestName = "wilcoxon",
) -> pd.DataFrame:
    """Build a deterministic significance table for all model pairs and properties."""
    if metric_proxy != "absolute_error":
        raise ValueError(
            f"Unsupported metric_proxy={metric_proxy!r}; only 'absolute_error' is implemented"
        )
    if not predictions_by_model:
        raise ValueError("At least one model prediction matrix is required")
    if y_true.ndim != 2:
        raise ValueError(f"y_true must be 2D [num_examples, num_properties], got {y_true.shape}")

    expected_shape = y_true.shape
    if expected_shape[1] != len(property_names):
        raise ValueError(
            f"property_names must match y_true width ({len(property_names)} != {expected_shape[1]})"
        )
    for model_name, predictions in predictions_by_model.items():
        if predictions.shape != expected_shape:
            raise ValueError(
                f"Prediction shape mismatch for model '{model_name}': "
                f"{predictions.shape} != {expected_shape}"
            )

    rows: list[dict[str, object]] = []
    sorted_models = sorted(predictions_by_model)
    for model_a, model_b in combinations(sorted_models, 2):
        preds_a = predictions_by_model[model_a]
        preds_b = predictions_by_model[model_b]
        for property_index, property_name in enumerate(property_names):
            targets = y_true[:, property_index]
            errors_a = np.abs(preds_a[:, property_index] - targets)
            errors_b = np.abs(preds_b[:, property_index] - targets)
            result = run_paired_significance_test(errors_a, errors_b, test_name=test_name)

            finite_mask = (
                np.isfinite(targets)
                & np.isfinite(preds_a[:, property_index])
                & np.isfinite(preds_b[:, property_index])
            )
            aligned_errors_a = errors_a[finite_mask]
            aligned_errors_b = errors_b[finite_mask]
            mean_error_a = (
                float(np.mean(aligned_errors_a)) if aligned_errors_a.size else float("nan")
            )
            mean_error_b = (
                float(np.mean(aligned_errors_b)) if aligned_errors_b.size else float("nan")
            )
            mean_delta = mean_error_a - mean_error_b
            winning_direction = _winning_direction(
                model_a=model_a,
                model_b=model_b,
                mean_error_a=mean_error_a,
                mean_error_b=mean_error_b,
                metric_proxy=metric_proxy,
            )

            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "property": property_name,
                    "metric_proxy": metric_proxy,
                    "sample_count": result.sample_count,
                    "test_name": result.test_name,
                    "statistic": result.statistic,
                    "p_value": result.p_value,
                    "mean_error_model_a": mean_error_a,
                    "mean_error_model_b": mean_error_b,
                    "mean_error_delta_a_minus_b": mean_delta,
                    "winning_direction": winning_direction,
                }
            )

    return pd.DataFrame(rows)


def _winning_direction(
    *,
    model_a: str,
    model_b: str,
    mean_error_a: float,
    mean_error_b: float,
    metric_proxy: str,
) -> str:
    if (
        math.isnan(mean_error_a)
        or math.isnan(mean_error_b)
        or np.isclose(mean_error_a, mean_error_b)
    ):
        return f"tie ({metric_proxy})"
    if mean_error_a < mean_error_b:
        return f"{model_a} lower {metric_proxy}"
    return f"{model_b} lower {metric_proxy}"


__all__ = [
    "PairedSignificanceResult",
    "PairedTestName",
    "build_pairwise_significance_table",
    "run_paired_significance_test",
]
