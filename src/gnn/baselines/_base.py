"""Shared implementation for multi-output tabular baseline regressors."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MultiOutputBaselineModel(ABC):
    """Base class for descriptor-based multi-output regressors."""

    def __init__(
        self,
        property_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.property_names = property_names or ["logS", "logP", "pKa"]
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.model_kwargs = dict(model_kwargs)
        self._models: dict[str, Any] = {}
        self._is_fitted = False
        self._n_features_in: int | None = None
        self._imputation_values: np.ndarray | None = None

    @abstractmethod
    def _create_model(self) -> Any:
        """Create an unfitted single-target model."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> MultiOutputBaselineModel:
        """Fit one model per target property."""
        X_array = self._validate_feature_matrix(X)
        y_array = self._validate_target_matrix(y)
        self._validate_property_names(y_array.shape[1])
        self._n_features_in = X_array.shape[1]
        self._ensure_feature_names(self._n_features_in)

        self._imputation_values = self._compute_imputation_values(X_array)
        X_train = self._apply_imputation(X_array)
        X_val_prepared = None if X_val is None else self._prepare_prediction_features(X_val)

        if y_val is not None:
            y_val_array = self._validate_target_matrix(y_val)
            if y_val_array.shape[1] != y_array.shape[1]:
                raise ValueError("y_val must have the same number of targets as y")
        else:
            y_val_array = None

        self._models = {}
        for index, property_name in enumerate(self.property_names):
            train_mask = ~np.isnan(y_array[:, index])
            if not np.any(train_mask):
                logger.warning("No valid labels for %s; skipping fit", property_name)
                continue

            model = self._create_model()
            val_targets = None if y_val_array is None else y_val_array[:, index]
            self._fit_single_target(
                model=model,
                X_train=X_train[train_mask],
                y_train=y_array[train_mask, index],
                X_val=X_val_prepared,
                y_val=val_targets,
            )
            self._models[property_name] = model

        self._is_fitted = True
        return self

    def _fit_single_target(
        self,
        *,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Fit a single-target model.

        Subclasses can override this to support algorithms with explicit
        validation-set behavior such as early stopping.
        """
        del X_val, y_val
        model.fit(X_train, y_train)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all configured target properties."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        prepared = self._prepare_prediction_features(X)
        predictions = np.full((prepared.shape[0], len(self.property_names)), np.nan, dtype=float)

        for index, property_name in enumerate(self.property_names):
            model = self._models.get(property_name)
            if model is not None:
                predictions[:, index] = model.predict(prepared)

        return predictions

    def get_feature_importances(self) -> pd.DataFrame:
        """Return a property-by-feature importance table."""
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        names = self.feature_names or [
            f"feature_{index}" for index in range(self._n_features_in or 0)
        ]
        importance_frame = pd.DataFrame(index=names)

        for property_name, model in self._models.items():
            importances = getattr(model, "feature_importances_", None)
            if importances is not None:
                importance_frame[property_name] = np.asarray(importances, dtype=float)

        if importance_frame.empty:
            raise RuntimeError("This model does not expose feature_importances_.")

        importance_frame["mean_importance"] = importance_frame.mean(axis=1)
        return importance_frame.sort_values("mean_importance", ascending=False)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model state with joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._serialize_state(), path)
        logger.info("Saved %s to %s", self.__class__.__name__, path)

    @classmethod
    def load(cls, path: str | Path) -> MultiOutputBaselineModel:
        """Load a model previously saved with :meth:`save`."""
        payload = joblib.load(path)
        instance = cls(
            property_names=payload["property_names"],
            feature_names=payload["feature_names"],
            **payload["model_kwargs"],
        )
        instance._restore_state(payload)
        return instance

    def _serialize_state(self) -> dict[str, Any]:
        return {
            "property_names": self.property_names,
            "feature_names": self.feature_names,
            "model_kwargs": self.model_kwargs,
            "models": self._models,
            "is_fitted": self._is_fitted,
            "n_features_in": self._n_features_in,
            "imputation_values": self._imputation_values,
        }

    def _restore_state(self, payload: dict[str, Any]) -> None:
        self._models = payload["models"]
        self._is_fitted = payload["is_fitted"]
        self._n_features_in = payload["n_features_in"]
        self._imputation_values = payload["imputation_values"]
        if self.feature_names is not None:
            self.moml_feature_names = list(self.feature_names)
        self.moml_property_names = list(self.property_names)
        self.moml_imputation_values = self._imputation_values

    def _validate_feature_matrix(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=float)
        if array.ndim != 2:
            raise ValueError("Expected a 2D feature matrix")
        if array.shape[0] == 0:
            raise ValueError("Feature matrix must contain at least one sample")
        return array

    def _validate_target_matrix(self, y: np.ndarray) -> np.ndarray:
        array = np.asarray(y, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("Expected a 1D or 2D target array")
        return array

    def _validate_property_names(self, num_targets: int) -> None:
        if len(self.property_names) != num_targets:
            raise ValueError(
                "property_names must match the number of target columns "
                f"({len(self.property_names)} != {num_targets})"
            )

    def _ensure_feature_names(self, num_features: int) -> None:
        if self.feature_names is None:
            self.feature_names = [f"feature_{index}" for index in range(num_features)]
            return
        if len(self.feature_names) != num_features:
            raise ValueError(
                "feature_names must match the number of feature columns "
                f"({len(self.feature_names)} != {num_features})"
            )

    def _compute_imputation_values(self, X: np.ndarray) -> np.ndarray:
        means = np.nanmean(X, axis=0)
        return np.where(np.isnan(means), 0.0, means)

    def _apply_imputation(self, X: np.ndarray) -> np.ndarray:
        if self._imputation_values is None:
            raise RuntimeError("Model imputation statistics are not initialized")
        prepared = np.asarray(X, dtype=float).copy()
        nan_rows, nan_cols = np.where(np.isnan(prepared))
        if len(nan_rows) > 0:
            prepared[nan_rows, nan_cols] = self._imputation_values[nan_cols]
        return prepared

    def _prepare_prediction_features(self, X: np.ndarray) -> np.ndarray:
        array = self._validate_feature_matrix(X)
        if self._n_features_in is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if array.shape[1] != self._n_features_in:
            raise ValueError(
                "Input feature dimension does not match training data "
                f"({array.shape[1]} != {self._n_features_in})"
            )
        return self._apply_imputation(array)


__all__ = ["MultiOutputBaselineModel"]
