"""Random Forest baseline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from gnn.baselines._base import MultiOutputBaselineModel

if TYPE_CHECKING:
    from sklearn.ensemble import RandomForestRegressor


class RandomForestBaseline(MultiOutputBaselineModel):
    """Backward-compatible wrapper retained for older multi-output workflows."""

    def __init__(
        self,
        property_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**self.model_kwargs)


def train_rf_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    property_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
    n_jobs: int = -1,
    **model_kwargs: Any,
) -> RandomForestBaseline | RandomForestRegressor:
    """Train a Random Forest baseline.

    Returns a native ``RandomForestRegressor`` for single-target or fully
    observed multi-target training. Sparse multi-target labels fall back to the
    project wrapper so each property can keep its partially observed rows.
    """
    from sklearn.ensemble import RandomForestRegressor

    X_array = _validate_feature_matrix(X_train)
    y_array = _validate_target_array(y_train)
    property_labels = _resolve_property_names(y_array, property_names)
    feature_labels = _resolve_feature_names(X_array.shape[1], feature_names)

    if y_array.shape[1] > 1 and np.isnan(y_array).any():
        model = RandomForestBaseline(
            property_names=property_labels,
            feature_names=feature_labels,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            **model_kwargs,
        )
        fitted = model.fit(X_array, y_array, X_val=X_val, y_val=y_val)
        fitted.moml_feature_names = feature_labels
        fitted.moml_property_names = property_labels
        fitted.moml_imputation_values = fitted._imputation_values
        return fitted

    valid_rows = ~np.any(np.isnan(y_array), axis=1)
    if not np.any(valid_rows):
        raise ValueError("Training targets must contain at least one fully observed row")

    imputation_values = _compute_imputation_values(X_array[valid_rows])
    X_prepared = _apply_imputation(X_array[valid_rows], imputation_values)
    y_prepared = y_array[valid_rows]

    target_data = y_prepared[:, 0] if y_prepared.shape[1] == 1 else y_prepared

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=n_jobs,
        **model_kwargs,
    )
    model.fit(X_prepared, target_data)
    model.moml_feature_names = feature_labels
    model.moml_property_names = property_labels
    model.moml_imputation_values = imputation_values
    return model


def predict_rf(model: RandomForestBaseline | RandomForestRegressor, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a fitted Random Forest baseline."""
    if isinstance(model, RandomForestBaseline):
        array = np.asarray(X, dtype=float)
        was_single_sample = False
        if array.ndim == 1:
            array = array.reshape(1, -1)
            was_single_sample = True
        predictions = np.asarray(model.predict(array), dtype=float)
        return predictions[0] if was_single_sample else predictions

    X_prepared, was_single_sample = _prepare_prediction_features(model, X)
    predictions = np.asarray(model.predict(X_prepared), dtype=float)

    if predictions.ndim == 1:
        return predictions[0] if was_single_sample else predictions
    return predictions[0] if was_single_sample else predictions


def save_rf_model(model: RandomForestBaseline | RandomForestRegressor, path: str | Path) -> None:
    """Persist a fitted Random Forest baseline."""
    if isinstance(model, RandomForestBaseline):
        model.save(path)
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_rf_model(path: str | Path) -> RandomForestBaseline | RandomForestRegressor:
    """Load a previously saved Random Forest baseline."""
    payload = joblib.load(path)
    if isinstance(payload, dict) and {"property_names", "models", "model_kwargs"} <= set(payload):
        return RandomForestBaseline.load(path)
    return payload


def get_feature_importance(
    model: RandomForestBaseline | RandomForestRegressor,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """Return feature importances sorted in descending order."""
    if isinstance(model, RandomForestBaseline):
        frame = model.get_feature_importances()
        if feature_names is not None:
            if len(feature_names) != len(frame.index):
                raise ValueError("feature_names must match the fitted feature dimension")
            frame = frame.copy()
            frame.index = feature_names
        return frame

    if not hasattr(model, "feature_importances_"):
        raise RuntimeError("Model is not fitted. Call train_rf_baseline() first.")

    names = feature_names or getattr(model, "moml_feature_names", None)
    if names is None:
        names = [f"feature_{index}" for index in range(int(model.n_features_in_))]
    if len(names) != int(model.n_features_in_):
        raise ValueError("feature_names must match the fitted feature dimension")

    frame = pd.DataFrame(
        {"importance": np.asarray(model.feature_importances_, dtype=float)},
        index=names,
    )
    return frame.sort_values("importance", ascending=False)


def get_rf_feature_importances(model: RandomForestBaseline | RandomForestRegressor) -> pd.DataFrame:
    """Backward-compatible alias for feature importance extraction."""
    return get_feature_importance(model)


def _validate_feature_matrix(X: np.ndarray) -> np.ndarray:
    array = np.asarray(X, dtype=float)
    if array.ndim != 2:
        raise ValueError("Expected a 2D feature matrix")
    if array.shape[0] == 0:
        raise ValueError("Feature matrix must contain at least one sample")
    return array


def _validate_target_array(y: np.ndarray) -> np.ndarray:
    array = np.asarray(y, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("Expected a 1D or 2D target array")
    return array


def _resolve_property_names(y: np.ndarray, property_names: list[str] | None) -> list[str]:
    num_targets = y.shape[1]
    if property_names is None:
        return [f"target_{index}" for index in range(num_targets)]
    if len(property_names) != num_targets:
        raise ValueError(
            "property_names must match the number of target columns "
            f"({len(property_names)} != {num_targets})"
        )
    return list(property_names)


def _resolve_feature_names(num_features: int, feature_names: list[str] | None) -> list[str]:
    if feature_names is None:
        return [f"feature_{index}" for index in range(num_features)]
    if len(feature_names) != num_features:
        raise ValueError(
            "feature_names must match the number of feature columns "
            f"({len(feature_names)} != {num_features})"
        )
    return list(feature_names)


def _compute_imputation_values(X: np.ndarray) -> np.ndarray:
    values = np.nanmean(X, axis=0)
    return np.where(np.isnan(values), 0.0, values)


def _apply_imputation(X: np.ndarray, imputation_values: np.ndarray) -> np.ndarray:
    prepared = np.asarray(X, dtype=float).copy()
    nan_rows, nan_cols = np.where(np.isnan(prepared))
    if len(nan_rows) > 0:
        prepared[nan_rows, nan_cols] = imputation_values[nan_cols]
    return prepared


def _prepare_prediction_features(
    model: RandomForestRegressor,
    X: np.ndarray,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(X, dtype=float)
    was_single_sample = False
    if array.ndim == 1:
        array = array.reshape(1, -1)
        was_single_sample = True
    elif array.ndim != 2:
        raise ValueError("Expected a 1D or 2D feature array")

    expected_features = int(model.n_features_in_)
    if array.shape[1] != expected_features:
        raise ValueError(
            "Input feature dimension does not match training data "
            f"({array.shape[1]} != {expected_features})"
        )

    imputation_values = getattr(model, "moml_imputation_values", None)
    if imputation_values is None:
        imputation_values = np.zeros(expected_features, dtype=float)

    return _apply_imputation(array, np.asarray(imputation_values, dtype=float)), was_single_sample


__all__ = [
    "RandomForestBaseline",
    "get_feature_importance",
    "get_rf_feature_importances",
    "load_rf_model",
    "predict_rf",
    "save_rf_model",
    "train_rf_baseline",
]
