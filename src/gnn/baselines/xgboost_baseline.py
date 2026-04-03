"""XGBoost baseline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd

from gnn.baselines._base import MultiOutputBaselineModel

if TYPE_CHECKING:
    from xgboost import XGBRegressor


class XGBoostBaseline(MultiOutputBaselineModel):
    """XGBoost baseline trained on descriptor features."""

    def __init__(
        self,
        property_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        eval_metric: str = "mae",
        early_stopping_rounds: int | None = 25,
        objective: str = "reg:squarederror",
        tree_method: str = "hist",
        verbosity: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            objective=objective,
            tree_method=tree_method,
            verbosity=verbosity,
            **kwargs,
        )

    def _create_model(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(**self.model_kwargs)

    def _fit_single_target(
        self,
        *,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        if (
            X_val is not None
            and y_val is not None
            and self.model_kwargs.get("early_stopping_rounds") is not None
        ):
            valid_mask = ~np.isnan(y_val)
            if np.any(valid_mask):
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val[valid_mask], y_val[valid_mask])],
                    verbose=False,
                )
                return
        model.fit(X_train, y_train, verbose=False)

    def save(self, path: str | Path) -> None:
        """Persist metadata with native XGBoost model files."""
        path = Path(path)
        base_dir = path if path.suffix == "" else path.with_suffix("")
        base_dir.mkdir(parents=True, exist_ok=True)

        payload = self._serialize_state()
        payload["models"] = list(self._models)
        payload["model_type"] = "wrapper"
        joblib.dump(payload, base_dir / "metadata.joblib")

        for property_name, model in self._models.items():
            model.get_booster().save_model(base_dir / f"{property_name}.json")

    @classmethod
    def load(cls, path: str | Path) -> XGBoostBaseline:
        """Restore metadata and native XGBoost model files."""
        path = Path(path)
        base_dir = path if path.suffix == "" else path.with_suffix("")
        payload = joblib.load(base_dir / "metadata.joblib")
        instance = cls(
            property_names=payload["property_names"],
            feature_names=payload["feature_names"],
            **payload["model_kwargs"],
        )
        instance._is_fitted = payload["is_fitted"]
        instance._n_features_in = payload["n_features_in"]
        instance._imputation_values = payload["imputation_values"]
        instance._models = {}

        for property_name in payload["models"]:
            model = instance._create_model()
            model.load_model(base_dir / f"{property_name}.json")
            instance._models[property_name] = model

        return instance


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    property_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    **model_kwargs: Any,
) -> XGBoostBaseline | XGBRegressor:
    """Train an XGBoost baseline model.

    Returns a native ``XGBRegressor`` for single-target or fully observed
    multi-target training. Sparse multi-target labels fall back to the project
    wrapper so each property can retain its partially observed rows.
    """
    y_array = _validate_target_array(y_train)
    y_val_array = None if y_val is None else _validate_target_array(y_val)

    if y_array.shape[1] > 1 and np.isnan(y_array).any():
        feature_labels = _resolve_feature_names(np.asarray(X_train).shape[1], feature_names)
        property_labels = property_names or [f"target_{index}" for index in range(y_array.shape[1])]
        if len(property_labels) != y_array.shape[1]:
            raise ValueError(
                "property_names must match the number of target columns "
                f"({len(property_labels)} != {y_array.shape[1]})"
            )

        model = XGBoostBaseline(
            property_names=list(property_labels),
            feature_names=feature_labels,
            **model_kwargs,
        )
        fitted = model.fit(X_train, y_array, X_val=X_val, y_val=y_val_array)
        fitted.moml_feature_names = feature_labels
        fitted.moml_property_names = list(property_labels)
        fitted.moml_imputation_values = fitted._imputation_values
        return fitted

    target_data = y_array[:, 0] if y_array.shape[1] == 1 else y_array
    val_targets = None
    if y_val_array is not None:
        val_targets = y_val_array[:, 0] if y_val_array.shape[1] == 1 else y_val_array

    return _train_native_xgb_model(
        X_train=X_train,
        y_train=target_data,
        X_val=X_val,
        y_val=val_targets,
        property_names=property_names,
        feature_names=feature_names,
        **model_kwargs,
    )


def train_xgb_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    property_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    **model_kwargs: Any,
) -> XGBoostBaseline | XGBRegressor:
    """Public alias for ``train_xgboost_baseline`` with the same return contract."""
    return train_xgboost_baseline(
        X_train,
        y_train,
        X_val=X_val,
        y_val=y_val,
        property_names=property_names,
        feature_names=feature_names,
        **model_kwargs,
    )


def predict_xgboost(model: XGBoostBaseline | XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a fitted XGBoost baseline."""
    if isinstance(model, XGBoostBaseline):
        array = np.asarray(X, dtype=float)
        was_single_sample = False
        if array.ndim == 1:
            array = array.reshape(1, -1)
            was_single_sample = True
        predictions = np.asarray(model.predict(array), dtype=float)
        return predictions[0] if was_single_sample else predictions

    X_prepared, was_single_sample = _prepare_native_prediction_features(model, X)
    predictions = np.asarray(model.predict(X_prepared), dtype=float)
    return predictions[0] if was_single_sample else predictions


def predict_xgb(model: XGBoostBaseline | XGBRegressor, X: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for ``predict_xgboost``."""
    return predict_xgboost(model, X)


def save_xgboost_model(model: XGBoostBaseline | XGBRegressor, path: str | Path) -> None:
    """Persist a fitted XGBoost baseline."""
    if isinstance(model, XGBoostBaseline):
        model.save(path)
        return
    _save_native_xgb_model(model, path)


def save_xgb_model(model: XGBoostBaseline | XGBRegressor, path: str | Path) -> None:
    """Backward-compatible alias for ``save_xgboost_model``."""
    save_xgboost_model(model, path)


def load_xgboost_model(path: str | Path) -> XGBoostBaseline | XGBRegressor:
    """Load a previously saved XGBoost baseline."""
    artifact_path = Path(path)
    metadata_path = _native_metadata_path(artifact_path)
    if metadata_path.exists():
        payload = joblib.load(metadata_path)
        if payload.get("model_type") == "native":
            return _load_native_xgb_model(artifact_path)
        return XGBoostBaseline.load(artifact_path)

    base_dir = _resolve_model_dir(artifact_path)
    payload = joblib.load(base_dir / "metadata.joblib")
    if payload.get("model_type") == "native":
        return _load_native_xgb_model(base_dir)
    return XGBoostBaseline.load(path)


def load_xgb_model(path: str | Path) -> XGBoostBaseline | XGBRegressor:
    """Backward-compatible alias for ``load_xgboost_model``."""
    return load_xgboost_model(path)


def get_xgboost_feature_importances(
    model: XGBoostBaseline | XGBRegressor,
) -> pd.DataFrame:
    """Backward-compatible feature-importance helper."""
    return get_feature_importance(model)


def get_feature_importance(
    model: XGBoostBaseline | XGBRegressor,
    feature_names: list[str] | None = None,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Return feature importances with configurable XGBoost importance types."""
    if importance_type not in {"weight", "gain", "cover"}:
        raise ValueError("importance_type must be one of: 'weight', 'gain', 'cover'")

    if isinstance(model, XGBoostBaseline):
        names = feature_names or model.feature_names
        if names is None:
            raise ValueError(
                "feature_names are required for multi-target XGBoost importance tables"
            )
        names = _validate_importance_feature_names(
            feature_names=list(names),
            expected_features=model._n_features_in,
        )
        columns: dict[str, np.ndarray] = {}
        for property_name, estimator in model._models.items():
            scores = estimator.get_booster().get_score(importance_type=importance_type)
            columns[property_name] = _score_vector(scores, names)
        frame = pd.DataFrame(columns, index=names)
        frame["mean_importance"] = frame.mean(axis=1)
        return frame.sort_values("mean_importance", ascending=False)

    names = feature_names or getattr(model, "moml_feature_names", None)
    if names is None:
        names = [f"feature_{index}" for index in range(int(model.n_features_in_))]
    names = _validate_importance_feature_names(
        feature_names=list(names),
        expected_features=int(model.n_features_in_),
    )
    scores = model.get_booster().get_score(importance_type=importance_type)
    frame = pd.DataFrame({"importance": _score_vector(scores, names)}, index=names)
    return frame.sort_values("importance", ascending=False)


def _train_native_xgb_model(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None,
    y_val: np.ndarray | None,
    property_names: list[str] | None,
    feature_names: list[str] | None,
    **model_kwargs: Any,
) -> XGBRegressor:
    from xgboost import XGBRegressor

    X_array = _validate_feature_matrix(X_train)
    y_array = np.asarray(y_train, dtype=float)
    if y_array.ndim == 1:
        valid_rows = ~np.isnan(y_array)
        num_targets = 1
    elif y_array.ndim == 2:
        valid_rows = ~np.any(np.isnan(y_array), axis=1)
        num_targets = y_array.shape[1]
    else:
        raise ValueError("Expected a 1D or 2D target array")

    if not np.any(valid_rows):
        raise ValueError("Training targets must contain at least one valid value")

    imputation_values = _compute_imputation_values(X_array[valid_rows])
    X_prepared = _apply_imputation(X_array[valid_rows], imputation_values)
    y_prepared = y_array[valid_rows]

    property_labels = property_names or [f"target_{index}" for index in range(num_targets)]
    if len(property_labels) != num_targets:
        raise ValueError(
            "property_names must match the number of target columns "
            f"({len(property_labels)} != {num_targets})"
        )
    feature_labels = _resolve_feature_names(X_array.shape[1], feature_names)

    fit_kwargs: dict[str, Any] = {}
    if (
        X_val is not None
        and y_val is not None
        and model_kwargs.get("early_stopping_rounds") is not None
    ):
        X_val_array = _validate_feature_matrix(X_val)
        y_val_array = np.asarray(y_val, dtype=float)
        if y_val_array.ndim == 1:
            valid_val = ~np.isnan(y_val_array)
            y_val_prepared = y_val_array[valid_val]
        elif y_val_array.ndim == 2:
            valid_val = ~np.any(np.isnan(y_val_array), axis=1)
            y_val_prepared = y_val_array[valid_val]
        else:
            raise ValueError("Expected a 1D or 2D validation target array")
        if np.any(valid_val):
            fit_kwargs["eval_set"] = [
                (
                    _apply_imputation(X_val_array[valid_val], imputation_values),
                    y_val_prepared,
                )
            ]
            fit_kwargs["verbose"] = False

    model = XGBRegressor(**model_kwargs)
    model.fit(X_prepared, y_prepared, **fit_kwargs)
    model.moml_feature_names = feature_labels
    model.moml_property_names = property_labels
    model.moml_imputation_values = imputation_values
    return model


def _save_native_xgb_model(model: XGBRegressor, path: str | Path) -> None:
    artifact_path = Path(path)
    model_path = _native_model_path(artifact_path)
    metadata_path = _native_metadata_path(artifact_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": "native",
        "feature_names": getattr(model, "moml_feature_names", None),
        "property_names": getattr(model, "moml_property_names", ["target_0"]),
        "imputation_values": getattr(model, "moml_imputation_values", None),
    }
    joblib.dump(metadata, metadata_path)
    model.get_booster().save_model(model_path)


def _load_native_xgb_model(path: str | Path) -> XGBRegressor:
    from xgboost import XGBRegressor

    artifact_path = Path(path)
    metadata = joblib.load(_native_metadata_path(artifact_path))
    model = XGBRegressor()
    model.load_model(_native_model_path(artifact_path))
    model.moml_feature_names = metadata.get("feature_names")
    model.moml_property_names = metadata.get("property_names")
    model.moml_imputation_values = metadata.get("imputation_values")
    return model


def _native_model_path(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.suffix else path_obj / "model.json"


def _native_metadata_path(path: str | Path) -> Path:
    path_obj = Path(path)
    if path_obj.suffix:
        return path_obj.with_name(f"{path_obj.name}.metadata.joblib")
    return path_obj / "metadata.joblib"


def _resolve_model_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.suffix == "" else path_obj.with_suffix("")


def _validate_importance_feature_names(
    *,
    feature_names: list[str],
    expected_features: int | None,
) -> list[str]:
    if expected_features is None:
        return feature_names
    if len(feature_names) != expected_features:
        raise ValueError(
            "feature_names must match the fitted feature dimension "
            f"({len(feature_names)} != {expected_features})"
        )
    return feature_names


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


def _prepare_native_prediction_features(
    model: XGBRegressor,
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


def _score_vector(scores: dict[str, float], feature_names: list[str]) -> np.ndarray:
    values = np.zeros(len(feature_names), dtype=float)
    for index, _ in enumerate(feature_names):
        values[index] = float(scores.get(f"f{index}", 0.0))
    return values


__all__ = [
    "XGBoostBaseline",
    "get_feature_importance",
    "get_xgboost_feature_importances",
    "load_xgb_model",
    "load_xgboost_model",
    "predict_xgb",
    "predict_xgboost",
    "save_xgb_model",
    "save_xgboost_model",
    "train_xgb_baseline",
    "train_xgboost_baseline",
]
