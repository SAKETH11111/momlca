"""Comparison utilities for baseline and GNN model outputs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gnn.baselines.protocol import Metadata, PredictorLike
from gnn.evaluation.confidence_intervals import flatten_confidence_interval_metadata
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
    predictor: PredictorLike
    model_type: str = "unknown"
    metadata: Metadata = field(default_factory=dict)


class ModelComparison:
    """Evaluate multiple models on the same splits and summarize metrics."""

    def __init__(
        self,
        property_names: list[str] | None = None,
        *,
        use_wandb: bool = True,
    ) -> None:
        self.property_names = property_names or ["logS", "logP", "pKa"]
        self.use_wandb = use_wandb
        self._models: dict[str, RegisteredModel] = {}
        self._results: list[ModelResult] = []

    def add_model(
        self,
        model_name: str,
        predictor: PredictorLike,
        model_type: str | None = None,
        metadata: Metadata | None = None,
    ) -> None:
        """Register a model for later evaluation."""
        self._models[model_name] = RegisteredModel(
            name=model_name,
            predictor=predictor,
            model_type=model_type or str((metadata or {}).get("model_type", "unknown")),
            metadata=metadata or {},
        )

    def evaluate(
        self,
        X: np.ndarray | None,
        y: np.ndarray,
        *,
        split_name: str = "test",
        dataset: Any | None = None,
        datamodule: Any | None = None,
    ) -> list[ModelResult]:
        """Evaluate all registered models on a single split."""
        if not self._models:
            raise RuntimeError("No models registered. Call add_model() first.")

        results = []
        for registered in self._models.values():
            predictions = self._predict_registered_model(
                registered.predictor,
                X=X,
                split_name=split_name,
                dataset=dataset,
                datamodule=datamodule,
            )
            results.append(
                self.add_result(
                    registered.name,
                    predictions,
                    y,
                    metadata={"model_type": registered.model_type, **registered.metadata},
                    split_name=split_name,
                )
            )
        return results

    def evaluate_splits(
        self,
        split_map: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> list[ModelResult]:
        """Evaluate all registered models across multiple named splits."""
        return self.evaluate_all_splits(dict(split_map))

    def evaluate_all_splits(
        self, split_map: dict[str, tuple[np.ndarray, np.ndarray] | dict[str, Any]]
    ) -> list[ModelResult]:
        """Evaluate all registered models across multiple named splits."""
        all_results: list[ModelResult] = []
        for split_name, split_input in split_map.items():
            X_split, y_split, dataset, datamodule = self._resolve_split_input(split_input)
            all_results.extend(
                self.evaluate(
                    X_split,
                    y_split,
                    split_name=split_name,
                    dataset=dataset,
                    datamodule=datamodule,
                )
            )
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
        filtered = [
            result
            for result in self._results
            if split_name is None or result.split_name == split_name
        ]
        records = []
        metadata_columns = sorted(
            {
                key
                for result in filtered
                for key, value in result.metadata.items()
                if key != "model_type" and self._is_scalar_metadata_value(value)
            }
        )
        ci_metadata_by_result: dict[int, dict[str, int | float | str | None]] = {
            index: self._confidence_interval_metadata(result.metadata)
            for index, result in enumerate(filtered)
        }
        ci_metadata_columns = sorted(
            {key for metadata in ci_metadata_by_result.values() for key in metadata}
        )
        for result_index, result in enumerate(filtered):
            record = {
                "model": result.model_name,
                "split": result.split_name,
                "model_type": result.metadata.get("model_type", "unknown"),
            }
            for key in metadata_columns:
                value = result.metadata.get(key)
                record[key] = value if self._is_scalar_metadata_value(value) else None
            for key in ci_metadata_columns:
                record[key] = ci_metadata_by_result[result_index].get(key)
            record.update(result.metrics)
            records.append(record)

        if not records:
            return pd.DataFrame()

        frame = pd.DataFrame(records)
        mean_columns = sorted(column for column in frame.columns if column.endswith("_mean"))
        metric_columns = sorted(
            column
            for column in frame.columns
            if column not in {"model", "split", "model_type", *metadata_columns}
            and not column.endswith("_mean")
        )
        ordered = [
            "model",
            "split",
            "model_type",
            *metadata_columns,
            *mean_columns,
            *metric_columns,
        ]
        frame = frame[ordered]

        if split_name is not None or frame["split"].nunique() == 1:
            return frame.set_index("model").drop(columns="split")

        return frame.set_index(["split", "model"])

    def _predict_registered_model(
        self,
        predictor: PredictorLike,
        *,
        X: np.ndarray | None,
        split_name: str,
        dataset: Any | None,
        datamodule: Any | None,
    ) -> np.ndarray:
        if hasattr(predictor, "predict_datamodule") and datamodule is not None:
            prediction = predictor.predict_datamodule(datamodule, split_name=split_name)
            return np.asarray(prediction, dtype=float)

        if hasattr(predictor, "predict_dataset") and dataset is not None:
            prediction = predictor.predict_dataset(dataset, split_name=split_name)
            return np.asarray(prediction, dtype=float)

        if hasattr(predictor, "predict"):
            if X is None:
                raise ValueError("predict(X) models require a feature matrix for evaluation")
            prediction = predictor.predict(X)
            return np.asarray(prediction, dtype=float)

        if hasattr(predictor, "predict_datamodule"):
            raise ValueError("predict_datamodule(...) models require a datamodule for evaluation")

        if hasattr(predictor, "predict_dataset"):
            raise ValueError("predict_dataset(...) models require a dataset for evaluation")

        raise TypeError(
            "Registered models must expose `predict(X)`, "
            "`predict_dataset(dataset, split_name=...)`, or "
            "`predict_datamodule(datamodule, split_name=...)`"
        )

    def _resolve_split_input(
        self,
        split_input: tuple[np.ndarray, np.ndarray] | dict[str, Any],
    ) -> tuple[np.ndarray | None, np.ndarray, Any | None, Any | None]:
        if isinstance(split_input, tuple):
            X_split, y_split = split_input
            return X_split, y_split, None, None

        if "y" in split_input:
            y_split = np.asarray(split_input["y"], dtype=float)
        elif "targets" in split_input:
            y_split = np.asarray(split_input["targets"], dtype=float)
        else:
            raise ValueError("Split dictionaries must include `y` or `targets`")

        feature_matrix: np.ndarray | None = split_input.get("X")
        if feature_matrix is not None:
            feature_matrix = np.asarray(feature_matrix, dtype=float)

        return (
            feature_matrix,
            y_split,
            split_input.get("dataset"),
            split_input.get("datamodule"),
        )

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
        significance: pd.DataFrame | None = None,
    ) -> None:
        """Write a Markdown report summarizing comparison results."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = self._filter_metric_columns(
            self.to_dataframe(split_name=split_name), metric_types
        )
        split_ranking_tables = self._build_split_rankings(summary, split_name=split_name)
        best_per_target = self._best_model_per_target(split_name=split_name)
        lines = [
            "# Baseline Model Comparison",
            "",
            f"Properties: {', '.join(self.property_names)}",
            "",
            "## Summary",
            "",
            summary.to_markdown(floatfmt=".4f"),
        ]

        if not summary.empty and "mae_mean" in summary.columns:
            best_model = self.get_best_model(
                metric="mae_mean", lower_is_better=True, split_name=split_name
            )
            lines.extend(["", "## Best Model", "", f"`{best_model}` has the lowest mean MAE."])

        if split_ranking_tables:
            lines.extend(["", "## Per-Split Rankings"])
            for name, table in split_ranking_tables.items():
                lines.extend(["", f"### {name}", "", table.to_markdown(floatfmt=".4f")])

        if best_per_target:
            lines.extend(["", "## Best Model Per Target", ""])
            for property_name, best_model in best_per_target.items():
                lines.append(f"- `{property_name}`: `{best_model}`")

        if significance is not None and not significance.empty:
            lines.extend(
                [
                    "",
                    "## Statistical Significance Results",
                    "",
                    significance.to_markdown(index=False, floatfmt=".6f"),
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    "## Statistical Significance Notes",
                    "",
                    "Statistical significance is not computed in this report yet.",
                    "Treat small metric gaps as directional unless they are confirmed with paired significance tests.",
                ]
            )

        output_path.write_text("\n".join(lines))

    def log_to_wandb(
        self,
        *,
        split_name: str | None = None,
        prefix: str = "baseline_comparison",
        run: Any | None = None,
    ) -> None:
        """Log comparison results to Weights & Biases when available."""
        if not self.use_wandb:
            logger.info("Skipping W&B logging because comparison is in offline mode")
            return
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
        columns = [
            column
            for column in frame.columns
            if any(column.startswith(metric) for metric in metric_types)
        ]
        passthrough = [column for column in frame.columns if column == "model_type"]
        return frame[[*passthrough, *columns]]

    def _build_split_rankings(
        self,
        frame: pd.DataFrame,
        *,
        split_name: str | None,
    ) -> dict[str, pd.DataFrame]:
        if frame.empty:
            return {}

        if split_name is not None or not isinstance(frame.index, pd.MultiIndex):
            working = frame.copy()
            if "mae_mean" in working.columns:
                working = working.sort_values("mae_mean", ascending=True)
            return {split_name or "overall": working}

        rankings: dict[str, pd.DataFrame] = {}
        for current_split in frame.index.get_level_values(0).unique():
            split_frame = frame.xs(current_split, level="split").copy()
            if "mae_mean" in split_frame.columns:
                split_frame = split_frame.sort_values("mae_mean", ascending=True)
            rankings[str(current_split)] = split_frame
        return rankings

    def _best_model_per_target(self, *, split_name: str | None) -> dict[str, str]:
        frame = self.to_dataframe(split_name=split_name)
        if frame.empty:
            return {}

        best_models: dict[str, str] = {}
        for property_name in self.property_names:
            metric_name = f"mae_{property_name}"
            if metric_name not in frame.columns:
                continue

            if isinstance(frame.index, pd.MultiIndex):
                summary = frame.groupby(level="model")[metric_name].mean()
                best_models[property_name] = str(summary.idxmin())
            else:
                best_models[property_name] = str(frame[metric_name].idxmin())

        return best_models

    @staticmethod
    def _is_scalar_metadata_value(value: object) -> bool:
        return isinstance(value, (str, int, float, bool, np.number))

    @staticmethod
    def _confidence_interval_metadata(metadata: Metadata) -> dict[str, int | float | str | None]:
        interval_map = metadata.get("confidence_intervals")
        if not isinstance(interval_map, Mapping):
            return {}
        typed_interval_map: dict[str, Mapping[str, Any]] = {
            str(metric_name): summary
            for metric_name, summary in interval_map.items()
            if isinstance(summary, Mapping)
        }
        return flatten_confidence_interval_metadata(typed_interval_map)


__all__ = ["ModelComparison", "ModelResult", "RegisteredModel"]
