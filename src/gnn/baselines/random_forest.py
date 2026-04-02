"""Random Forest baseline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gnn.baselines._base import MultiOutputBaselineModel


class RandomForestBaseline(MultiOutputBaselineModel):
    """Random Forest baseline trained on descriptor features."""

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
    **model_kwargs: Any,
) -> RandomForestBaseline:
    """Train and return a Random Forest baseline model."""
    model = RandomForestBaseline(
        property_names=property_names,
        feature_names=feature_names,
        **model_kwargs,
    )
    return model.fit(X_train, y_train, X_val=X_val, y_val=y_val)


def predict_rf(model: RandomForestBaseline, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a fitted Random Forest baseline."""
    return model.predict(X)


def save_rf_model(model: RandomForestBaseline, path: str | Path) -> None:
    """Persist a fitted Random Forest baseline."""
    model.save(path)


def load_rf_model(path: str | Path) -> RandomForestBaseline:
    """Load a previously saved Random Forest baseline."""
    return RandomForestBaseline.load(path)


def get_rf_feature_importances(model: RandomForestBaseline) -> Any:
    """Return property-wise feature importances."""
    return model.get_feature_importances()


__all__ = [
    "RandomForestBaseline",
    "get_rf_feature_importances",
    "load_rf_model",
    "predict_rf",
    "save_rf_model",
    "train_rf_baseline",
]
