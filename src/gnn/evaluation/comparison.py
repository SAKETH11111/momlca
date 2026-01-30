"""Model comparison framework for evaluating multiple models.

Provides utilities to compare GNN and ML baseline models on the same
test data with consistent metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results from a single model evaluation.

    Attributes:
        model_name: Identifier for the model
        predictions: Model predictions (n_samples, n_properties)
        targets: Ground truth targets (n_samples, n_properties)
        property_names: Names of properties predicted
        metrics: Computed metrics dict
        metadata: Additional model info (hyperparams, etc.)
    """

    model_name: str
    predictions: np.ndarray
    targets: np.ndarray
    property_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    property_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute regression metrics for predictions.

    Computes MAE, RMSE, R², and Spearman correlation for each property
    and averaged across properties.

    Args:
        y_true: Ground truth of shape (n_samples,) or (n_samples, n_properties)
        y_pred: Predictions of shape (n_samples,) or (n_samples, n_properties)
        property_names: Names of properties (auto-generated if None)

    Returns:
        Dictionary of metric_name -> value with keys like:
        - mae_<prop>, rmse_<prop>, r2_<prop>, spearman_<prop> for each property
        - mae_mean, rmse_mean, r2_mean, spearman_mean for averages
    """
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_properties = y_true.shape[1]
    property_names = property_names or [f"prop_{i}" for i in range(n_properties)]

    metrics: dict[str, float] = {}

    for i, prop_name in enumerate(property_names):
        y_t = y_true[:, i]
        y_p = y_pred[:, i]

        # Mask NaN values
        valid_mask = ~np.isnan(y_t) & ~np.isnan(y_p)
        y_t_valid = y_t[valid_mask]
        y_p_valid = y_p[valid_mask]

        if len(y_t_valid) == 0:
            logger.warning("No valid samples for property %s, skipping metrics", prop_name)
            continue

        # MAE
        mae = float(np.mean(np.abs(y_t_valid - y_p_valid)))
        metrics[f"mae_{prop_name}"] = mae

        # RMSE
        rmse = float(np.sqrt(np.mean((y_t_valid - y_p_valid) ** 2)))
        metrics[f"rmse_{prop_name}"] = rmse

        # R²
        ss_res = np.sum((y_t_valid - y_p_valid) ** 2)
        ss_tot = np.sum((y_t_valid - np.mean(y_t_valid)) ** 2)
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
        metrics[f"r2_{prop_name}"] = r2

        # Spearman correlation
        if len(y_t_valid) > 2:
            spearman_corr, _ = spearmanr(y_t_valid, y_p_valid)
            metrics[f"spearman_{prop_name}"] = float(spearman_corr)

    # Compute averages across properties
    for metric_type in ["mae", "rmse", "r2", "spearman"]:
        prop_values = [v for k, v in metrics.items() if k.startswith(f"{metric_type}_")]
        if prop_values:
            metrics[f"{metric_type}_mean"] = float(np.mean(prop_values))

    return metrics


class ModelComparison:
    """Compare multiple models on the same evaluation data.

    Collects results from different models and generates comparison tables.

    Example:
        >>> comparison = ModelComparison(property_names=["logS", "logP", "pKa"])
        >>> comparison.add_result("RandomForest", rf_predictions, y_test)
        >>> comparison.add_result("XGBoost", xgb_predictions, y_test)
        >>> comparison.add_result("GNN", gnn_predictions, y_test)
        >>> print(comparison.to_table())
    """

    def __init__(
        self,
        property_names: list[str] | None = None,
    ) -> None:
        """Initialize ModelComparison.

        Args:
            property_names: Names of target properties
        """
        self.property_names = property_names or ["logS", "logP", "pKa"]
        self._results: list[ModelResult] = []

    def add_result(
        self,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> ModelResult:
        """Add model evaluation result.

        Args:
            model_name: Identifier for the model
            predictions: Model predictions
            targets: Ground truth targets
            metadata: Additional model info

        Returns:
            ModelResult with computed metrics
        """
        metrics = compute_regression_metrics(targets, predictions, self.property_names)

        result = ModelResult(
            model_name=model_name,
            predictions=predictions,
            targets=targets,
            property_names=self.property_names,
            metrics=metrics,
            metadata=metadata or {},
        )

        self._results.append(result)
        logger.info(
            "Added result for %s: MAE_mean=%.4f, R2_mean=%.4f",
            model_name,
            metrics.get("mae_mean", float("nan")),
            metrics.get("r2_mean", float("nan")),
        )

        return result

    @property
    def results(self) -> list[ModelResult]:
        """Get all collected results."""
        return list(self._results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame.

        Returns:
            DataFrame with models as rows and metrics as columns
        """
        records = []
        for result in self._results:
            record = {"model": result.model_name}
            record.update(result.metrics)
            records.append(record)

        df = pd.DataFrame(records)
        if len(df) == 0:
            return df

        df = df.set_index("model")

        # Reorder columns: mean metrics first, then per-property
        mean_cols = [c for c in df.columns if c.endswith("_mean")]
        other_cols = [c for c in df.columns if not c.endswith("_mean")]
        df = df[sorted(mean_cols) + sorted(other_cols)]

        return df

    def to_table(self, metric_types: list[str] | None = None) -> str:
        """Generate formatted comparison table.

        Args:
            metric_types: Which metric types to include (e.g., ["mae", "r2"])
                If None, includes all metrics.

        Returns:
            Formatted string table
        """
        df = self.to_dataframe()

        if metric_types:
            cols = [c for c in df.columns if any(c.startswith(m) for m in metric_types)]
            df = df[cols]

        # Format floats
        return df.to_string(float_format=lambda x: f"{x:.4f}")

    def to_latex(self, metric_types: list[str] | None = None) -> str:
        """Generate LaTeX table for publication.

        Args:
            metric_types: Which metric types to include

        Returns:
            LaTeX table string
        """
        df = self.to_dataframe()

        if metric_types:
            cols = [c for c in df.columns if any(c.startswith(m) for m in metric_types)]
            df = df[cols]

        return df.to_latex(float_format=lambda x: f"{x:.3f}")

    def save(self, path: str) -> None:
        """Save comparison results to CSV.

        Args:
            path: Path to save CSV file
        """
        df = self.to_dataframe()
        df.to_csv(path)
        logger.info("Saved comparison results to %s", path)

    def get_best_model(self, metric: str = "mae_mean", lower_is_better: bool = True) -> str:
        """Get the name of the best performing model.

        Args:
            metric: Metric to use for comparison
            lower_is_better: Whether lower metric values are better

        Returns:
            Name of the best model
        """
        df = self.to_dataframe()
        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found. Available: {list(df.columns)}")

        if lower_is_better:
            return str(df[metric].idxmin())
        else:
            return str(df[metric].idxmax())
