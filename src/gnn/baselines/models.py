"""Traditional ML baseline models for molecular property prediction.

Provides scikit-learn compatible models (Random Forest, XGBoost) that
can be trained on molecular descriptors.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@runtime_checkable
class BaselineModel(Protocol):
    """Protocol defining the interface for baseline models."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineModel":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    def load(cls, path: str | Path) -> "BaselineModel":
        ...


class MultiOutputBaselineModel(ABC):
    """Base class for multi-output regression baseline models.

    Handles the common pattern of training one model per property
    when the underlying algorithm doesn't natively support multi-output.

    Args:
        property_names: Names of target properties
        **model_kwargs: Passed to the underlying model constructor
    """

    def __init__(
        self,
        property_names: list[str] | None = None,
        **model_kwargs: Any,
    ) -> None:
        self.property_names = property_names or ["logS", "logP", "pKa"]
        self.model_kwargs = model_kwargs
        self._models: dict[str, Any] = {}
        self._is_fitted = False

    @abstractmethod
    def _create_model(self) -> Any:
        """Create a single model instance."""
        ...

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiOutputBaselineModel":
        """Fit models for each target property.

        Args:
            X: Features of shape (n_samples, n_features)
            y: Targets of shape (n_samples, n_properties)

        Returns:
            Self for method chaining
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for i, prop_name in enumerate(self.property_names):
            logger.info("Training model for %s", prop_name)

            # Get targets for this property
            y_prop = y[:, i]

            # Mask NaN values
            valid_mask = ~np.isnan(y_prop)
            X_valid = X[valid_mask]
            y_valid = y_prop[valid_mask]

            if len(y_valid) == 0:
                logger.warning("No valid samples for %s, skipping", prop_name)
                continue

            # Handle NaN in features by imputing with column means
            X_valid = self._impute_nan_features(X_valid)

            # Create and fit model
            model = self._create_model()
            model.fit(X_valid, y_valid)
            self._models[prop_name] = model

        self._is_fitted = True
        return self

    def _impute_nan_features(self, X: np.ndarray) -> np.ndarray:
        """Impute NaN values in features with column means."""
        X = X.copy()
        col_means = np.nanmean(X, axis=0)
        # Replace NaN column means with 0
        col_means = np.where(np.isnan(col_means), 0, col_means)

        for col in range(X.shape[1]):
            nan_mask = np.isnan(X[:, col])
            X[nan_mask, col] = col_means[col]

        return X

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict all target properties.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples, n_properties)
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Impute NaN features
        X = self._impute_nan_features(X)

        n_samples = X.shape[0]
        predictions = np.full((n_samples, len(self.property_names)), np.nan)

        for i, prop_name in enumerate(self.property_names):
            if prop_name in self._models:
                predictions[:, i] = self._models[prop_name].predict(X)

        return predictions

    def save(self, path: str | Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "models": self._models,
                "property_names": self.property_names,
                "model_kwargs": self.model_kwargs,
                "is_fitted": self._is_fitted,
                "class_name": self.__class__.__name__,
            },
            path,
        )
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "MultiOutputBaselineModel":
        """Load model from disk."""
        data = joblib.load(path)
        instance = cls(property_names=data["property_names"], **data["model_kwargs"])
        instance._models = data["models"]
        instance._is_fitted = data["is_fitted"]
        return instance


class RandomForestBaseline(MultiOutputBaselineModel):
    """Random Forest baseline for molecular property prediction.

    Args:
        property_names: Names of target properties
        n_estimators: Number of trees in the forest
        max_depth: Maximum depth of trees (None for unlimited)
        min_samples_split: Minimum samples to split a node
        n_jobs: Number of parallel jobs (-1 for all cores)
        random_state: Random seed for reproducibility

    Example:
        >>> from gnn.baselines import RandomForestBaseline, MolecularDescriptorExtractor
        >>> extractor = MolecularDescriptorExtractor()
        >>> X = extractor.fit_transform(smiles_list)
        >>> model = RandomForestBaseline(n_estimators=100)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        property_names: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            property_names=property_names,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self) -> Any:
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**self.model_kwargs)


class XGBoostBaseline(MultiOutputBaselineModel):
    """XGBoost baseline for molecular property prediction.

    Args:
        property_names: Names of target properties
        n_estimators: Number of boosting rounds
        max_depth: Maximum tree depth
        learning_rate: Boosting learning rate
        subsample: Subsample ratio of training data
        colsample_bytree: Subsample ratio of columns
        random_state: Random seed for reproducibility

    Example:
        >>> from gnn.baselines import XGBoostBaseline
        >>> model = XGBoostBaseline(n_estimators=100, max_depth=6)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        property_names: list[str] | None = None,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            property_names=property_names,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            **kwargs,
        )

    def _create_model(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(**self.model_kwargs)
