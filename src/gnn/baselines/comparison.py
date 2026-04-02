"""Comparison utilities for baseline and GNN model outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn.baselines.protocol import Metadata, SupportsPredict
from gnn.evaluation.metrics.regression import compute_regression_metrics

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results from evaluating a single model on one split."""

    model_name: str
    split_name: str
    predictions: np.ndarray
    targets: np.ndarray
    property_names: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass
class RegisteredModel:
    """Model registered for later comparison."""

    name: str
    predictor: SupportsPredict
    metadata: Metadata = field(default_factory=dict)


class ModelComparison:
    """Evaluate multiple models on the same splits and summarize metrics."""

    def __init__(self, property_names: list[str] | None = None) -> None:
        self.property_names = property_names or ["logS", "logP", "pKa"]
        self._models: dict[str, RegisteredModel] = {}
        self._results: list[ModelResult] = []

    def add_model(
        self,
        model_name: str,
        predictor: SupportsPredict,
        metadata: Metadata | None = None,
    ) -> None:
        """Register a model for later evaluation."""
        self._models[model_name] = RegisteredModel(
            name=model_name,
            predictor=predictor,
            metadata=metadata or {},
        )

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        split_name: str = "test",
    ) -> list[ModelResult]:
        """Evaluate all registered models on a single split."""
        if not self._models:
            raise RuntimeError("No models registered. Call add_model() first.")

        results = []
        for registered in self._models.values():
            predictions = registered.predictor.predict(X)
            results.append(
                self.add_result(
                    registered.name,
                    predictions,
                    y,
                    metadata=registered.metadata,
                    split_name=split_name,
                )
            )
        return results

    def evaluate_splits(
        self,
        split_map: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> list[ModelResult]:
        """Evaluate all registered models across multiple named splits."""
        all_results: list[ModelResult] = []
        for split_name, (X_split, y_split) in split_map.items():
            all_results.extend(self.evaluate(X_split, y_split, split_name=split_name))
        return all_results

    def add_result(
        self,
        model_name: str,
        predictions: np.ndarray,
        targets: np.ndarray,
        metadata: Metadata | None = None,
        split_name: str = "test",
    ) -> ModelResult:
        """Add a precomputed prediction result.

        This compatibility method keeps the original workflow working while the
        preferred path is now ``add_model()`` followed by ``evaluate()``.
        """
        metrics = compute_regression_metrics(targets, predictions, self.property_names)
        result = ModelResult(
            model_name=model_name,
            split_name=split_name,
            predictions=np.asarray(predictions, dtype=float),
            targets=np.asarray(targets, dtype=float),
            property_names=list(self.property_names),
            metrics=metrics,
            metadata=metadata or {},
        )
        self._results.append(result)
        return result

    @property
    def results(self) -> list[ModelResult]:
        """Return all accumulated comparison results."""
        return list(self._results)

    def to_dataframe(self, split_name: str | None = None) -> pd.DataFrame:
        """Convert evaluation results to a DataFrame."""
        filtered = [result for result in self._results if split_name is None or result.split_name == split_name]
        records = []
        for result in filtered:
            record = {
                "model": result.model_name,
                "split": result.split_name,
            }
            record.update(result.metrics)
            records.append(record)

        if not records:
            return pd.DataFrame()

        frame = pd.DataFrame(records)
        mean_columns = sorted(column for column in frame.columns if column.endswith("_mean"))
        metric_columns = sorted(
            column
            for column in frame.columns
            if column not in {"model", "split"} and not column.endswith("_mean")
        )
        ordered = ["model", "split", *mean_columns, *metric_columns]
        frame = frame[ordered]

        if split_name is not None or frame["split"].nunique() == 1:
            return frame.set_index("model").drop(columns="split")

        return frame.set_index(["split", "model"])

    def to_table(
        self,
        metric_types: list[str] | None = None,
        *,
        split_name: str | None = None,
    ) -> str:
        """Return a plain-text comparison table."""
        frame = self._filter_metric_columns(self.to_dataframe(split_name=split_name), metric_types)
        return frame.to_string(float_format=lambda value: f"{value:.4f}")

    def to_markdown(
        self,
        metric_types: list[str] | None = None,
        *,
        split_name: str | None = None,
    ) -> str:
        """Return a Markdown comparison table."""
        frame = self._filter_metric_columns(self.to_dataframe(split_name=split_name), metric_types)
        return frame.to_markdown(floatfmt=".4f")

    def to_latex(
        self,
        metric_types: list[str] | None = None,
        *,
        split_name: str | None = None,
    ) -> str:
        """Return a LaTeX comparison table."""
        frame = self._filter_metric_columns(self.to_dataframe(split_name=split_name), metric_types)
        return frame.to_latex(float_format=lambda value: f"{value:.3f}")

    def save(self, path: str | Path, *, split_name: str | None = None) -> None:
        """Persist comparison metrics to CSV."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe(split_name=split_name).to_csv(output_path)

    def save_report(
        self,
        path: str | Path,
        *,
        split_name: str | None = None,
        metric_types: list[str] | None = None,
    ) -> None:
        """Write a Markdown report summarizing comparison results."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self._filter_metric_columns(self.to_dataframe(split_name=split_name), metric_types)
        lines = [
            "# Baseline Model Comparison",
            "",
            f"Properties: {', '.join(self.property_names)}",
            "",
            "## Metrics",
            "",
            summary.to_markdown(floatfmt=".4f"),
        ]

        if not summary.empty and "mae_mean" in summary.columns:
            best_model = self.get_best_model(metric="mae_mean", lower_is_better=True, split_name=split_name)
            lines.extend(["", "## Best Model", "", f"`{best_model}` has the lowest mean MAE."])

        output_path.write_text("\n".join(lines))

    def log_to_wandb(
        self,
        *,
        split_name: str | None = None,
        prefix: str = "baseline_comparison",
        run: Any | None = None,
    ) -> None:
        """Log comparison results to Weights & Biases when available."""
        frame = self.to_dataframe(split_name=split_name)
        if frame.empty:
            logger.warning("Skipping W&B logging because no comparison results were recorded")
            return

        if run is None:
            import wandb

            if wandb.run is None:
                logger.warning("Skipping W&B logging because no active run exists")
                return
            run = wandb.run
        else:
            import wandb

        table = frame.reset_index()
        run.log(
            {
                f"{prefix}/table": wandb.Table(dataframe=table),
            }
        )

    def get_best_model(
        self,
        metric: str = "mae_mean",
        lower_is_better: bool = True,
        *,
        split_name: str | None = None,
    ) -> str:
        """Return the best model according to the requested metric."""
        frame = self.to_dataframe(split_name=split_name)
        if metric not in frame.columns:
            raise ValueError(f"Metric {metric} not found. Available: {list(frame.columns)}")
        if lower_is_better:
            return str(frame[metric].idxmin())
        return str(frame[metric].idxmax())

    def _filter_metric_columns(
        self,
        frame: pd.DataFrame,
        metric_types: list[str] | None,
    ) -> pd.DataFrame:
        if frame.empty or metric_types is None:
            return frame
        columns = [column for column in frame.columns if any(column.startswith(metric) for metric in metric_types)]
        return frame[columns]


__all__ = ["ModelComparison", "ModelResult", "RegisteredModel"]
