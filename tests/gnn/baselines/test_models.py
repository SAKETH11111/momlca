"""Tests for Random Forest and XGBoost baseline models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from gnn.baselines.random_forest import (
    load_rf_model,
    predict_rf,
    save_rf_model,
    train_rf_baseline,
)
from gnn.baselines.xgboost_baseline import (
    XGBoostBaseline,
    load_xgboost_model,
    predict_xgboost,
    save_xgboost_model,
    train_xgboost_baseline,
)


@pytest.fixture
def regression_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Create an easy multi-output regression problem."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(150, 12))
    y = np.column_stack(
        [
            np.where(X[:, 0] > 0, 1.5, -1.5) + 0.25 * X[:, 1],
            X[:, 2] ** 2 - 0.4 * X[:, 3],
            np.sin(X[:, 4]) + np.where(X[:, 5] > 0, X[:, 5], -0.2 * X[:, 5]),
        ]
    )
    y += rng.normal(scale=0.02, size=y.shape)

    y[3, 0] = np.nan
    y[10, 2] = np.nan
    X[5, 1] = np.nan
    X[8, 6] = np.nan

    feature_names = [f"feature_{index}" for index in range(X.shape[1])]
    property_names = ["logS", "logP", "pKa"]

    return X[:110], y[:110], X[110:], y[110:], feature_names, property_names


class TestRandomForestBaseline:
    def test_function_api_and_prediction_shape(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, X_test, _, feature_names, property_names = regression_data

        model = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=120,
            random_state=42,
        )
        predictions = predict_rf(model, X_test)

        assert predictions.shape == (40, 3)
        assert model.feature_names == feature_names

    def test_feature_importances_include_feature_names(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, _, _, feature_names, property_names = regression_data
        model = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=80,
            random_state=1,
        )

        importance_frame = model.get_feature_importances()

        assert "mean_importance" in importance_frame.columns
        assert importance_frame.index[0] in feature_names

    def test_save_and_load_round_trip(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, X_test, _, feature_names, property_names = regression_data
        model = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=60,
            random_state=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rf.joblib"
            save_rf_model(model, path)
            loaded = load_rf_model(path)

            np.testing.assert_allclose(model.predict(X_test), loaded.predict(X_test))

    def test_input_dimension_validation(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, _, _, feature_names, property_names = regression_data
        model = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=40,
            random_state=0,
        )

        with pytest.raises(ValueError, match="dimension"):
            model.predict(np.ones((2, 5)))

    def test_reaches_low_mae_on_easy_problem(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, X_test, y_test, feature_names, property_names = regression_data
        model = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=200,
            random_state=11,
        )
        predictions = model.predict(X_test)

        mae = np.nanmean(np.abs(predictions - y_test))
        assert mae < 0.5


class TestXGBoostBaseline:
    def test_function_api_supports_validation_split(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, X_val, y_val, feature_names, property_names = regression_data
        model = train_xgboost_baseline(
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=120,
            early_stopping_rounds=10,
            random_state=42,
        )

        predictions = predict_xgboost(model, X_val)
        assert predictions.shape == (40, 3)

    def test_native_save_and_load_round_trip(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, X_test, _, feature_names, property_names = regression_data
        model = train_xgboost_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=80,
            early_stopping_rounds=None,
            random_state=8,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "xgb_model"
            save_xgboost_model(model, path)
            loaded = load_xgboost_model(path)

            np.testing.assert_allclose(model.predict(X_test), loaded.predict(X_test))

    def test_feature_importances_available(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, _, _, feature_names, property_names = regression_data
        model = train_xgboost_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=60,
            early_stopping_rounds=None,
            random_state=13,
        )

        importance_frame = model.get_feature_importances()
        assert importance_frame.index[0] in feature_names

    def test_predict_dimension_validation(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, _, _, feature_names, property_names = regression_data
        model = XGBoostBaseline(
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=40,
            early_stopping_rounds=None,
        ).fit(X_train, y_train)

        with pytest.raises(ValueError, match="dimension"):
            model.predict(np.ones((3, 2)))

    def test_reaches_low_mae_on_easy_problem(
        self,
        regression_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]],
    ) -> None:
        X_train, y_train, X_test, y_test, feature_names, property_names = regression_data
        model = train_xgboost_baseline(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=120,
            early_stopping_rounds=10,
            random_state=5,
        )

        predictions = model.predict(X_test)
        mae = np.nanmean(np.abs(predictions - y_test))
        assert mae < 0.35
