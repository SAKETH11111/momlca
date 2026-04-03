"""Compatibility exports for baseline model implementations."""

from gnn.baselines._base import MultiOutputBaselineModel
from gnn.baselines.protocol import BaselineModel
from gnn.baselines.random_forest import RandomForestBaseline
from gnn.baselines.xgboost_baseline import XGBoostBaseline

__all__ = [
    "BaselineModel",
    "MultiOutputBaselineModel",
    "RandomForestBaseline",
    "XGBoostBaseline",
]
