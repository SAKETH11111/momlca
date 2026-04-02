"""Protocols shared by tabular baseline models and comparison utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class BaselineModel(Protocol):
    """Protocol for trainable tabular regressors."""

    property_names: list[str]
    feature_names: list[str] | None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> BaselineModel:
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str | Path) -> None:
        ...

    @classmethod
    def load(cls, path: str | Path) -> BaselineModel:
        ...

    def get_feature_importances(self) -> pd.DataFrame:
        ...


class SupportsPredict(Protocol):
    """Protocol for objects that can generate tabular predictions."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


Metadata = dict[str, Any]

__all__ = ["BaselineModel", "Metadata", "SupportsPredict"]
