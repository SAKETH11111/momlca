"""Tests for model comparison framework."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gnn.evaluation.comparison import (
    ModelComparison,
    ModelResult,
    compute_regression_metrics,
)


class TestComputeRegressionMetrics:
    """Test suite for compute_regression_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_pred = y_true.copy()

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1", "prop2", "prop3"])

        assert metrics["mae_prop1"] == 0.0
        assert metrics["mae_prop2"] == 0.0
        assert metrics["mae_prop3"] == 0.0
        assert metrics["mae_mean"] == 0.0
        assert metrics["rmse_mean"] == 0.0
        assert np.isclose(metrics["r2_mean"], 1.0)

    def test_known_error(self) -> None:
        """Test metrics with known error values."""
        y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_pred = np.array([[0.5], [1.5], [2.5], [3.5]])  # All off by 0.5

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1"])

        assert np.isclose(metrics["mae_prop1"], 0.5)
        assert np.isclose(metrics["rmse_prop1"], 0.5)

    def test_handles_nan_targets(self) -> None:
        """Test that NaN targets are properly masked."""
        y_true = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
        y_pred = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0]])

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1", "prop2"])

        # prop1: only indices 0, 2 are valid -> MAE = 0
        assert metrics["mae_prop1"] == 0.0
        # prop2: only indices 0, 1 are valid -> MAE = 0
        assert metrics["mae_prop2"] == 0.0

    def test_1d_input(self) -> None:
        """Test that 1D inputs are handled correctly."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert "mae_prop_0" in metrics
        assert metrics["mae_prop_0"] == 0.0

    def test_spearman_correlation(self) -> None:
        """Test Spearman correlation calculation."""
        # Perfect monotonic relationship
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1"])

        assert np.isclose(metrics["spearman_prop1"], 1.0)


class TestModelComparison:
    """Test suite for ModelComparison class."""

    def test_initialization(self) -> None:
        """Test comparison initialization."""
        comparison = ModelComparison(property_names=["logS", "logP", "pKa"])
        assert comparison.property_names == ["logS", "logP", "pKa"]
        assert len(comparison.results) == 0

    def test_add_result(self) -> None:
        """Test adding model results."""
        comparison = ModelComparison(property_names=["prop1", "prop2"])

        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1]])

        result = comparison.add_result("Model1", y_pred, y_true)

        assert isinstance(result, ModelResult)
        assert result.model_name == "Model1"
        assert "mae_prop1" in result.metrics
        assert len(comparison.results) == 1

    def test_multiple_models(self) -> None:
        """Test comparing multiple models."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0], [3.0]])

        # Model 1: Good predictions
        comparison.add_result("Good", np.array([[1.0], [2.0], [3.0]]), y_true)

        # Model 2: Poor predictions
        comparison.add_result("Bad", np.array([[0.0], [0.0], [0.0]]), y_true)

        assert len(comparison.results) == 2

        df = comparison.to_dataframe()
        assert len(df) == 2
        assert "Good" in df.index
        assert "Bad" in df.index

        # Good model should have lower MAE
        assert df.loc["Good", "mae_prop1"] < df.loc["Bad", "mae_prop1"]

    def test_to_table(self) -> None:
        """Test table output generation."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [2.0]])

        comparison.add_result("Model1", y_pred, y_true)

        table = comparison.to_table()
        assert "Model1" in table
        assert "mae" in table.lower()

    def test_to_table_filter_metrics(self) -> None:
        """Test filtering metrics in table output."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [2.0]])

        comparison.add_result("Model1", y_pred, y_true)

        table = comparison.to_table(metric_types=["mae", "r2"])
        assert "mae" in table.lower()
        assert "r2" in table.lower()

    def test_save(self) -> None:
        """Test saving comparison to CSV."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [2.0]])

        comparison.add_result("Model1", y_pred, y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.csv"
            comparison.save(str(path))

            assert path.exists()
            content = path.read_text()
            assert "Model1" in content

    def test_get_best_model(self) -> None:
        """Test getting best model."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0], [3.0]])

        # Good model (lower MAE)
        comparison.add_result("Good", np.array([[1.0], [2.0], [3.0]]), y_true)
        # Bad model (higher MAE)
        comparison.add_result("Bad", np.array([[0.0], [0.0], [0.0]]), y_true)

        best = comparison.get_best_model(metric="mae_mean", lower_is_better=True)
        assert best == "Good"

        # For R2, higher is better
        best_r2 = comparison.get_best_model(metric="r2_mean", lower_is_better=False)
        assert best_r2 == "Good"

    def test_get_best_model_invalid_metric(self) -> None:
        """Test that invalid metric raises error."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        comparison.add_result("Model1", y_true, y_true)

        with pytest.raises(ValueError, match="not found"):
            comparison.get_best_model(metric="invalid_metric")

    def test_empty_comparison_dataframe(self) -> None:
        """Test that empty comparison returns empty dataframe."""
        comparison = ModelComparison(property_names=["prop1"])
        df = comparison.to_dataframe()
        assert len(df) == 0

    def test_metadata_stored(self) -> None:
        """Test that metadata is stored in results."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])

        result = comparison.add_result(
            "Model1",
            y_true,
            y_true,
            metadata={"n_estimators": 100, "max_depth": 5},
        )

        assert result.metadata["n_estimators"] == 100
        assert result.metadata["max_depth"] == 5
