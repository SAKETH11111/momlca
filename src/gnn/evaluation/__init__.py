"""Evaluation utilities for molecular property prediction.

This module provides tools for evaluating and comparing model performance
on regression tasks.
"""

from gnn.evaluation.comparison import (
    ModelComparison,
    ModelResult,
    compute_regression_metrics,
)

__all__ = [
    "ModelComparison",
    "ModelResult",
    "compute_regression_metrics",
]
