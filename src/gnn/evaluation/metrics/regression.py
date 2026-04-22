"""Regression metrics shared by baseline and GNN evaluation code."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Literal, cast

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

NaNPolicy = Literal["omit", "raise"]


def compute_regression_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_pred: np.ndarray | torch.Tensor,
    property_names: list[str] | None = None,
    *,
    nan_policy: NaNPolicy = "omit",
) -> dict[str, float]:
    """Compute regression metrics per property and across-property means.

    Args:
        y_true: Target values as a NumPy array or torch tensor.
        y_pred: Predicted values as a NumPy array or torch tensor.
        property_names: Optional names for each regression target.
        nan_policy: ``"omit"`` to mask NaN pairs per property (default) or
            ``"raise"`` to fail if NaNs are present in either input.
    """
    if nan_policy not in {"omit", "raise"}:
        raise ValueError(f"Unsupported nan_policy={nan_policy!r}; expected 'omit' or 'raise'.")

    true_values = _to_numpy_array(y_true)
    pred_values = _to_numpy_array(y_pred)

    if true_values.ndim == 1:
        true_values = true_values.reshape(-1, 1)
    if pred_values.ndim == 1:
        pred_values = pred_values.reshape(-1, 1)

    if true_values.shape != pred_values.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape "
            f"({true_values.shape} != {pred_values.shape})"
        )

    num_properties = true_values.shape[1]
    property_labels = property_names or [f"prop_{index}" for index in range(num_properties)]
    if len(property_labels) != num_properties:
        raise ValueError("property_names must match the number of target columns")

    if nan_policy == "raise" and (np.isnan(true_values).any() or np.isnan(pred_values).any()):
        raise ValueError(
            "NaN values detected in y_true or y_pred while nan_policy='raise'. "
            "Use nan_policy='omit' to ignore NaN pairs."
        )

    metrics: dict[str, float] = {}

    for index, property_name in enumerate(property_labels):
        targets = true_values[:, index]
        predictions = pred_values[:, index]
        if nan_policy == "omit":
            valid_mask = ~np.isnan(targets) & ~np.isnan(predictions)
            targets = targets[valid_mask]
            predictions = predictions[valid_mask]

        if len(targets) == 0:
            continue

        errors = targets - predictions
        metrics[f"mae_{property_name}"] = float(np.mean(np.abs(errors)))
        metrics[f"rmse_{property_name}"] = float(np.sqrt(np.mean(errors**2)))

        centered_targets = targets - np.mean(targets)
        denominator = float(np.sum(centered_targets**2))
        if denominator == 0.0:
            metrics[f"r2_{property_name}"] = 1.0 if np.allclose(targets, predictions) else 0.0
        else:
            metrics[f"r2_{property_name}"] = float(1.0 - (np.sum(errors**2) / denominator))

        metrics[f"pearson_{property_name}"] = _safe_correlation(pearsonr, targets, predictions)
        metrics[f"spearman_{property_name}"] = _safe_correlation(spearmanr, targets, predictions)

    for metric_prefix in ("mae", "rmse", "r2", "pearson", "spearman"):
        values = [value for key, value in metrics.items() if key.startswith(f"{metric_prefix}_")]
        finite_values = [value for value in values if not np.isnan(value)]
        if finite_values:
            metrics[f"{metric_prefix}_mean"] = float(np.mean(finite_values))

    return metrics


def _to_numpy_array(values: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(float, copy=False)
    return np.asarray(values, dtype=float)


def _safe_correlation(
    correlation_fn: Callable[[np.ndarray, np.ndarray], object],
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = cast(tuple[float, float], correlation_fn(x, y))
    return float(result[0])


__all__ = ["NaNPolicy", "compute_regression_metrics"]
