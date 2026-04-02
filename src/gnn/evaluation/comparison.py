"""Compatibility exports for comparison utilities."""

from gnn.baselines.comparison import ModelComparison, ModelResult
from gnn.evaluation.metrics.regression import compute_regression_metrics

__all__ = ["ModelComparison", "ModelResult", "compute_regression_metrics"]
