"""Tests for baseline ML models."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from gnn.baselines.models import RandomForestBaseline, XGBoostBaseline


@pytest.fixture
def sample_data() -> tuple[np.ndarray, np.ndarray]:
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    n_properties = 3

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples, n_properties)
    # Add some NaN targets
    y[5, 0] = np.nan
    y[10, 1] = np.nan

    return X, y


class TestRandomForestBaseline:
    """Test suite for RandomForestBaseline."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        model = RandomForestBaseline(n_estimators=10)
        assert model.property_names == ["logS", "logP", "pKa"]
        assert not model._is_fitted

    def test_initialization_custom_properties(self) -> None:
        """Test model with custom property names."""
        model = RandomForestBaseline(
            property_names=["prop1", "prop2"],
            n_estimators=10,
        )
        assert model.property_names == ["prop1", "prop2"]

    def test_fit_predict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test training and prediction."""
        X, y = sample_data
        model = RandomForestBaseline(n_estimators=10)

        model.fit(X, y)

        assert model._is_fitted
        assert len(model._models) == 3

        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_save_load(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test model persistence."""
        X, y = sample_data
        model = RandomForestBaseline(n_estimators=10)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            model.save(path)

            loaded = RandomForestBaseline.load(path)

            # Check predictions match
            original_preds = model.predict(X)
            loaded_preds = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)

    def test_predict_without_fit_raises(self) -> None:
        """Test that predict raises if not fitted."""
        model = RandomForestBaseline()
        X = np.random.randn(10, 50)

        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(X)

    def test_handles_nan_features(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test that model handles NaN features via imputation."""
        X, y = sample_data
        # Add NaN features
        X[0, 0] = np.nan
        X[5, 10] = np.nan

        model = RandomForestBaseline(n_estimators=10)
        model.fit(X, y)

        predictions = model.predict(X)
        assert predictions.shape == y.shape
        # Predictions should not be NaN
        assert not np.any(np.isnan(predictions))


class TestXGBoostBaseline:
    """Test suite for XGBoostBaseline."""

    def test_initialization(self) -> None:
        """Test model initialization."""
        model = XGBoostBaseline(n_estimators=10, max_depth=3)
        assert model.model_kwargs["max_depth"] == 3
        assert model.model_kwargs["n_estimators"] == 10

    def test_fit_predict(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test training and prediction."""
        X, y = sample_data
        model = XGBoostBaseline(n_estimators=10)

        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == y.shape

    def test_save_load(self, sample_data: tuple[np.ndarray, np.ndarray]) -> None:
        """Test model persistence."""
        X, y = sample_data
        model = XGBoostBaseline(n_estimators=10)
        model.fit(X, y)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            model.save(path)

            loaded = XGBoostBaseline.load(path)

            # Check predictions match
            original_preds = model.predict(X)
            loaded_preds = loaded.predict(X)
            np.testing.assert_array_almost_equal(original_preds, loaded_preds)
