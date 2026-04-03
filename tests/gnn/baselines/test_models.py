"""Tests for Random Forest and XGBoost baseline models."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("xgboost")

from gnn.baselines.random_forest import (
    RandomForestBaseline,
    load_rf_model,
    predict_rf,
    save_rf_model,
    train_rf_baseline,
)
from gnn.baselines.random_forest import (
    get_feature_importance as get_rf_feature_importance,
)
from gnn.baselines.xgboost_baseline import (
    XGBoostBaseline,
    load_xgb_model,
    predict_xgb,
    predict_xgboost,
    save_xgb_model,
    train_xgb_baseline,
    train_xgboost_baseline,
)
from gnn.baselines.xgboost_baseline import (
    get_feature_importance as get_xgb_feature_importance,
)


@pytest.fixture
def regression_data() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
]:
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
    def test_single_target_returns_native_random_forest_regressor(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        from sklearn.ensemble import RandomForestRegressor

        X_train, y_train, X_test, _, feature_names, _ = regression_data
        model = train_rf_baseline(
            X_train,
            y_train[:, 0],
            property_names=["logS"],
            feature_names=feature_names,
            n_estimators=80,
            random_state=42,
        )

        assert isinstance(model, RandomForestRegressor)
        prediction = predict_rf(model, X_test[0])
        assert np.isscalar(prediction)

    def test_function_api_and_prediction_shape(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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
        assert model.moml_feature_names == feature_names
        assert isinstance(model, RandomForestBaseline)

    def test_feature_importances_include_feature_names(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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

        importance_frame = get_rf_feature_importance(model, feature_names)

        assert importance_frame.index[0] in feature_names
        assert "mean_importance" in importance_frame.columns

    def test_save_and_load_round_trip(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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

            np.testing.assert_allclose(predict_rf(model, X_test), predict_rf(loaded, X_test))
            assert loaded.moml_feature_names == feature_names

    def test_input_dimension_validation(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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
            predict_rf(model, np.ones((2, 5)))

    def test_single_sample_prediction(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_test, _, feature_names, property_names = regression_data
        model = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            random_state=9,
        )

        prediction = predict_rf(model, X_test[0])
        assert prediction.shape == (3,)

    def test_reaches_low_mae_on_easy_problem(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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

    def test_random_state_reproducibility(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_test, _, feature_names, property_names = regression_data
        model_one = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=120,
            random_state=123,
        )
        model_two = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=120,
            random_state=123,
        )

        np.testing.assert_allclose(predict_rf(model_one, X_test), predict_rf(model_two, X_test))

    def test_sparse_multi_target_uses_wrapper_to_keep_partial_labels(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_test, _, feature_names, property_names = regression_data

        model = train_rf_baseline(
            X_train,
            y_train,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=50,
            random_state=101,
        )

        assert isinstance(model, RandomForestBaseline)
        predictions = predict_rf(model, X_test)
        assert predictions.shape == (40, 3)


class TestXGBoostBaseline:
    def test_function_api_supports_validation_split(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_val, y_val, feature_names, property_names = regression_data
        model = train_xgb_baseline(
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

        predictions = predict_xgb(model, X_val)
        assert predictions.shape == (40, 3)
        assert isinstance(model, XGBoostBaseline)

    def test_native_save_and_load_round_trip(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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
            save_xgb_model(model, path)
            loaded = load_xgb_model(path)

            np.testing.assert_allclose(
                predict_xgboost(model, X_test), predict_xgboost(loaded, X_test)
            )

    def test_native_save_honors_requested_json_file_path(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_test, y_test, feature_names, _ = regression_data
        model = train_xgb_baseline(
            X_train,
            y_train[:, 0],
            X_val=X_test,
            y_val=y_test[:, 0],
            property_names=["logS"],
            feature_names=feature_names,
            n_estimators=20,
            early_stopping_rounds=None,
            random_state=12,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.json"
            save_xgb_model(model, path)

            assert path.exists()
            assert (Path(tmpdir) / "model.json.metadata.joblib").exists()

            loaded = load_xgb_model(path)
            np.testing.assert_allclose(predict_xgb(model, X_test), predict_xgb(loaded, X_test))

    def test_feature_importances_available(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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

        importance_frame = get_xgb_feature_importance(
            model,
            feature_names=feature_names,
            importance_type="gain",
        )
        assert importance_frame.index[0] in feature_names
        assert "mean_importance" in importance_frame.columns

    def test_feature_importance_rejects_mismatched_feature_names(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, _, _, feature_names, _ = regression_data
        model = train_xgb_baseline(
            X_train,
            y_train[:, 0],
            property_names=["logS"],
            feature_names=feature_names,
            n_estimators=20,
            early_stopping_rounds=None,
            random_state=5,
        )

        with pytest.raises(ValueError, match="feature_names must match"):
            get_xgb_feature_importance(model, feature_names=["a", "b"], importance_type="gain")

    def test_predict_dimension_validation(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, _, _, feature_names, property_names = regression_data
        model = XGBoostBaseline(
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=40,
            early_stopping_rounds=None,
        ).fit(X_train, y_train)

        with pytest.raises(ValueError, match="dimension"):
            predict_xgboost(model, np.ones((3, 2)))

    def test_reaches_low_mae_on_easy_problem(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
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

    def test_single_sample_prediction_support(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_test, y_test, feature_names, property_names = regression_data
        model = train_xgboost_baseline(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=60,
            early_stopping_rounds=5,
            random_state=21,
        )

        prediction = predict_xgb(model, X_test[0])
        assert prediction.shape == (3,)

    def test_single_target_alias_returns_native_xgb_regressor(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        from xgboost import XGBRegressor

        X_train, y_train, X_test, y_test, feature_names, _ = regression_data
        model = train_xgb_baseline(
            X_train,
            y_train[:, 0],
            X_val=X_test,
            y_val=y_test[:, 0],
            property_names=["logS"],
            feature_names=feature_names,
            n_estimators=50,
            early_stopping_rounds=5,
            random_state=31,
        )

        assert isinstance(model, XGBRegressor)
        prediction = predict_xgb(model, X_test[0])
        assert np.isscalar(prediction)

    def test_reproducibility_and_early_stopping(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_test, y_test, feature_names, _ = regression_data
        model_one = train_xgb_baseline(
            X_train,
            y_train[:, 0],
            X_val=X_test,
            y_val=y_test[:, 0],
            property_names=["logS"],
            feature_names=feature_names,
            n_estimators=120,
            early_stopping_rounds=10,
            random_state=77,
        )
        model_two = train_xgb_baseline(
            X_train,
            y_train[:, 0],
            X_val=X_test,
            y_val=y_test[:, 0],
            property_names=["logS"],
            feature_names=feature_names,
            n_estimators=120,
            early_stopping_rounds=10,
            random_state=77,
        )

        np.testing.assert_allclose(predict_xgb(model_one, X_test), predict_xgb(model_two, X_test))
        assert getattr(model_one, "best_iteration", 0) >= 0

    def test_sparse_multi_target_uses_wrapper_to_keep_partial_labels(
        self,
        regression_data: tuple[
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]
        ],
    ) -> None:
        X_train, y_train, X_test, y_test, feature_names, property_names = regression_data

        model = train_xgb_baseline(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test,
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=60,
            early_stopping_rounds=5,
            random_state=17,
        )

        assert isinstance(model, XGBoostBaseline)
        predictions = predict_xgb(model, X_test)
        assert predictions.shape == (40, 3)
