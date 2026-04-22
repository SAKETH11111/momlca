"""Focused tests for shared regression metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gnn.evaluation.metrics import compute_regression_metrics


def test_compute_regression_metrics_known_values() -> None:
    """MAE/RMSE/R2 should match hand-computed values for simple inputs."""
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_pred = np.array([[0.5], [1.5], [2.5], [3.5]])

    metrics = compute_regression_metrics(y_true, y_pred, ["prop"])

    assert np.isclose(metrics["mae_prop"], 0.5)
    assert np.isclose(metrics["rmse_prop"], 0.5)
    assert np.isclose(metrics["r2_prop"], 0.8)
    assert np.isclose(metrics["mae_mean"], 0.5)
    assert np.isclose(metrics["rmse_mean"], 0.5)
    assert np.isclose(metrics["r2_mean"], 0.8)


def test_compute_regression_metrics_supports_1d_single_target_inputs() -> None:
    """Single-target 1D arrays should be accepted without caller reshaping."""
    y_true = np.array([1.0, 2.0, 3.0], dtype=float)
    y_pred = np.array([1.0, 2.5, 3.5], dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred)

    assert "mae_prop_0" in metrics
    assert np.isclose(metrics["mae_prop_0"], (0.0 + 0.5 + 0.5) / 3.0)


def test_compute_regression_metrics_tensor_numpy_parity() -> None:
    """Torch and NumPy inputs with same numbers should produce equal metrics."""
    y_true_np = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=float)
    y_pred_np = np.array([[1.1, 2.1], [1.9, 3.7], [3.2, 6.4]], dtype=float)

    numpy_metrics = compute_regression_metrics(y_true_np, y_pred_np, ["a", "b"])
    tensor_metrics = compute_regression_metrics(
        torch.tensor(y_true_np, dtype=torch.float32),
        torch.tensor(y_pred_np, dtype=torch.float32),
        ["a", "b"],
    )

    assert set(numpy_metrics) == set(tensor_metrics)
    for key in numpy_metrics:
        np_value = numpy_metrics[key]
        torch_value = tensor_metrics[key]
        if np.isnan(np_value):
            assert np.isnan(torch_value)
        else:
            assert np.isclose(np_value, torch_value, rtol=1e-6, atol=1e-8)


def test_compute_regression_metrics_omits_nan_pairs_by_default() -> None:
    """Default NaN behavior should mask invalid rows per property."""
    y_true = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
    y_pred = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0]])

    metrics = compute_regression_metrics(y_true, y_pred, ["left", "right"])

    assert metrics["mae_left"] == 0.0
    assert metrics["mae_right"] == 0.0
    assert metrics["mae_mean"] == 0.0


def test_compute_regression_metrics_strict_nan_policy_raises() -> None:
    """Strict mode should fail loudly when NaNs are present."""
    y_true = np.array([[1.0], [np.nan]])
    y_pred = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="NaN"):
        compute_regression_metrics(y_true, y_pred, ["prop"], nan_policy="raise")


def test_compute_regression_metrics_constant_targets_keep_nan_correlations() -> None:
    """Undefined correlations should remain NaN and not create poisoned means."""
    y_true = np.array([[2.0], [2.0], [2.0]], dtype=float)
    y_pred = np.array([[1.0], [2.0], [3.0]], dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred, ["prop"])

    assert np.isnan(metrics["pearson_prop"])
    assert np.isnan(metrics["spearman_prop"])
    assert "pearson_mean" not in metrics
    assert "spearman_mean" not in metrics


def test_compute_regression_metrics_fully_masked_property_skips_means() -> None:
    """Properties with no valid pairs should not contribute metric keys or means."""
    y_true = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=float)
    y_pred = np.array([[1.0, 5.0], [2.0, 6.0]], dtype=float)

    metrics = compute_regression_metrics(y_true, y_pred, ["kept", "masked"])

    assert "mae_kept" in metrics
    assert "mae_masked" not in metrics
    assert np.isclose(metrics["mae_mean"], metrics["mae_kept"])


def test_compute_regression_metrics_rejects_invalid_nan_policy() -> None:
    """Unknown NaN policies should be rejected clearly."""
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[1.0], [2.0]])

    with pytest.raises(ValueError, match="nan_policy"):
        compute_regression_metrics(y_true, y_pred, ["prop"], nan_policy="invalid")
