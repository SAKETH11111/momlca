"""Checkpoint ablation orchestration built on canonical evaluation exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict

from gnn.baselines.comparison import ModelComparison
from gnn.evaluation.export import checkpoint_export_id, resolve_prediction_export_path
from gnn.evaluation.significance import PairedTestName, build_pairwise_significance_table


@dataclass(frozen=True)
class PredictionExport:
    """Loaded prediction export artifact for one model checkpoint."""

    model_name: str
    source_path: Path
    split_name: str
    checkpoint_path: str
    property_names: list[str]
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class AlignedPredictionData:
    """Aligned per-example targets/predictions for model comparison."""

    split_name: str
    property_names: list[str]
    sample_keys: list[tuple[str, str, str]]
    y_true: np.ndarray
    predictions_by_model: dict[str, np.ndarray]
    checkpoint_paths: dict[str, str]


@dataclass(frozen=True)
class AblationComparisonArtifacts:
    """Output artifacts produced by one ablation comparison run."""

    comparison_csv: Path
    significance_csv: Path
    report_md: Path
    run_id: str
    comparison_df: pd.DataFrame
    significance_df: pd.DataFrame


def load_prediction_export(path: str | Path, *, model_name: str | None = None) -> PredictionExport:
    """Load and validate one exported evaluation record JSON payload."""
    source_path = Path(path)
    payload = json.loads(source_path.read_text())
    metadata = payload.get("metadata")
    records = payload.get("records")
    if not isinstance(metadata, dict):
        raise ValueError(f"Prediction export metadata missing or invalid in {source_path}")
    if not isinstance(records, list):
        raise ValueError(f"Prediction export records missing or invalid in {source_path}")

    split_name = str(metadata.get("split") or "")
    checkpoint_path = str(metadata.get("checkpoint_path") or "")
    if split_name == "":
        raise ValueError(f"Prediction export split missing in {source_path}")
    if checkpoint_path == "":
        raise ValueError(f"Prediction export checkpoint_path missing in {source_path}")

    names_from_metadata = metadata.get("property_names")
    property_names: list[str] = []
    if isinstance(names_from_metadata, list):
        property_names = [str(name) for name in names_from_metadata]
    if not property_names and records:
        first_record = records[0]
        if isinstance(first_record, dict):
            prediction_map = first_record.get("predictions")
            if isinstance(prediction_map, dict):
                property_names = [str(name) for name in prediction_map]

    if not property_names:
        raise ValueError(f"Unable to determine property_names for export {source_path}")

    normalized_records = [dict(record) for record in records if isinstance(record, dict)]
    if len(normalized_records) != len(records):
        raise ValueError(f"Prediction export includes non-object record entries in {source_path}")

    resolved_name = model_name or source_path.stem
    return PredictionExport(
        model_name=resolved_name,
        source_path=source_path,
        split_name=split_name,
        checkpoint_path=checkpoint_path,
        property_names=property_names,
        records=normalized_records,
    )


def evaluate_checkpoints_with_eval_config(
    *,
    checkpoint_paths: dict[str, str],
    eval_cfg: DictConfig,
    prediction_output_dir: str | Path,
) -> dict[str, Path]:
    """Evaluate checkpoints via canonical eval pipeline and return export paths."""
    output_dir = Path(prediction_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    export_paths: dict[str, Path] = {}
    for model_name, checkpoint_path in checkpoint_paths.items():
        from src.eval import evaluate

        run_cfg = OmegaConf.create(OmegaConf.to_container(eval_cfg, resolve=False))
        with open_dict(run_cfg):
            run_cfg.ckpt_path = checkpoint_path
            run_cfg.export_predictions = True
            run_cfg.prediction_split = "test"
            run_cfg.log_prediction_artifact = False
            run_cfg.prediction_export_dir = str(output_dir)

        evaluate(run_cfg)
        export_path = resolve_prediction_export_path(
            output_dir=output_dir,
            split_name="test",
            checkpoint_path=checkpoint_path,
        )
        if not export_path.exists():
            raise FileNotFoundError(
                f"Evaluation completed but export file was not found for {model_name}: {export_path}"
            )
        export_paths[model_name] = export_path

    return export_paths


def align_prediction_exports(
    exports: dict[str, PredictionExport],
) -> AlignedPredictionData:
    """Align exported prediction rows by stable sample identifiers and properties."""
    if len(exports) < 2:
        raise ValueError("Ablation comparison requires at least two exports")

    ordered_models = sorted(exports)
    reference_model = ordered_models[0]
    reference_export = exports[reference_model]
    split_name = reference_export.split_name
    property_names = list(reference_export.property_names)

    for model_name in ordered_models[1:]:
        current = exports[model_name]
        if current.split_name != split_name:
            raise ValueError(
                f"Prediction split mismatch between '{reference_model}' ({split_name}) and "
                f"'{model_name}' ({current.split_name})"
            )
        if list(current.property_names) != property_names:
            raise ValueError(
                f"Property mismatch between '{reference_model}' and '{model_name}': "
                f"{property_names} != {current.property_names}"
            )

    record_maps = {
        model_name: _record_map_for_export(export, property_names)
        for model_name, export in exports.items()
    }
    reference_keys = set(record_maps[reference_model])
    for model_name in ordered_models[1:]:
        current_keys = set(record_maps[model_name])
        if current_keys != reference_keys:
            missing = sorted(reference_keys - current_keys)
            extra = sorted(current_keys - reference_keys)
            raise ValueError(
                f"Exports are not safely aligned for '{model_name}': "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )

    ordered_keys = sorted(reference_keys)
    y_true = np.full((len(ordered_keys), len(property_names)), np.nan, dtype=float)
    predictions_by_model: dict[str, np.ndarray] = {
        model_name: np.full((len(ordered_keys), len(property_names)), np.nan, dtype=float)
        for model_name in ordered_models
    }

    for row_index, key in enumerate(ordered_keys):
        reference_targets, reference_predictions = record_maps[reference_model][key]
        y_true[row_index, :] = reference_targets
        predictions_by_model[reference_model][row_index, :] = reference_predictions
        for model_name in ordered_models[1:]:
            targets, predictions = record_maps[model_name][key]
            if not np.allclose(targets, reference_targets, equal_nan=True):
                raise ValueError(
                    f"Target mismatch for aligned sample {key} between "
                    f"'{reference_model}' and '{model_name}'"
                )
            predictions_by_model[model_name][row_index, :] = predictions

    return AlignedPredictionData(
        split_name=split_name,
        property_names=property_names,
        sample_keys=ordered_keys,
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        checkpoint_paths={
            model_name: exports[model_name].checkpoint_path for model_name in ordered_models
        },
    )


def run_ablation_comparison(
    exports: dict[str, PredictionExport],
    *,
    output_dir: str | Path,
    significance_test: PairedTestName = "wilcoxon",
    use_wandb: bool = False,
) -> AblationComparisonArtifacts:
    """Generate deterministic ablation comparison artifacts from aligned exports."""
    aligned = align_prediction_exports(exports)
    comparison = ModelComparison(property_names=aligned.property_names, use_wandb=use_wandb)

    for model_name in sorted(aligned.predictions_by_model):
        checkpoint_path = aligned.checkpoint_paths[model_name]
        comparison.add_result(
            model_name,
            aligned.predictions_by_model[model_name],
            aligned.y_true,
            split_name=aligned.split_name,
            metadata={
                "model_type": "gnn",
                "checkpoint_path": checkpoint_path,
                "checkpoint_id": checkpoint_export_id(checkpoint_path),
            },
        )

    significance_df = build_pairwise_significance_table(
        y_true=aligned.y_true,
        predictions_by_model=aligned.predictions_by_model,
        property_names=aligned.property_names,
        test_name=significance_test,
    )
    comparison_df = comparison.to_dataframe(split_name=aligned.split_name)
    run_id = _ablation_run_id(aligned.checkpoint_paths)

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    stem = f"ablation-{aligned.split_name}-{run_id}"
    comparison_csv = output_dir_path / f"{stem}-comparison.csv"
    significance_csv = output_dir_path / f"{stem}-significance.csv"
    report_md = output_dir_path / f"{stem}-report.md"

    comparison.save(comparison_csv, split_name=aligned.split_name)
    significance_df.to_csv(significance_csv, index=False)
    comparison.save_report(
        report_md,
        split_name=aligned.split_name,
        significance=significance_df,
    )

    return AblationComparisonArtifacts(
        comparison_csv=comparison_csv,
        significance_csv=significance_csv,
        report_md=report_md,
        run_id=run_id,
        comparison_df=comparison_df,
        significance_df=significance_df,
    )


def _record_map_for_export(
    export: PredictionExport,
    property_names: list[str],
) -> dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]]:
    record_map: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for record in export.records:
        record_split = str(record.get("split", ""))
        if record_split != export.split_name:
            raise ValueError(
                f"Record split mismatch in {export.source_path}: "
                f"expected '{export.split_name}', found '{record_split}'"
            )
        key = _record_key(record)
        if key in record_map:
            raise ValueError(f"Duplicate sample key {key} in {export.source_path}")
        targets = _values_for_properties(
            mapping=record.get("targets"),
            property_names=property_names,
            field_name="targets",
            source=str(export.source_path),
        )
        predictions = _values_for_properties(
            mapping=record.get("predictions"),
            property_names=property_names,
            field_name="predictions",
            source=str(export.source_path),
        )
        record_map[key] = (targets, predictions)
    return record_map


def _values_for_properties(
    *,
    mapping: Any,
    property_names: list[str],
    field_name: str,
    source: str,
) -> np.ndarray:
    if not isinstance(mapping, dict):
        raise ValueError(f"Record field '{field_name}' must be a mapping in {source}")
    values = np.full(len(property_names), np.nan, dtype=float)
    for index, property_name in enumerate(property_names):
        if property_name not in mapping:
            raise ValueError(
                f"Record field '{field_name}' missing property '{property_name}' in {source}"
            )
        values[index] = _coerce_float(mapping[property_name])
    return values


def _record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    inchikey = str(record.get("inchikey") or "").strip()
    smiles = str(record.get("smiles") or "").strip()
    name = str(record.get("name") or "").strip()
    if inchikey == "" and smiles == "":
        raise ValueError("Record alignment requires at least inchikey or smiles")
    return (inchikey, smiles, name)


def _coerce_float(value: Any) -> float:
    if value is None:
        return float("nan")
    return float(value)


def _ablation_run_id(checkpoint_paths: dict[str, str]) -> str:
    digest_source = "|".join(
        f"{model_name}:{checkpoint_paths[model_name]}" for model_name in sorted(checkpoint_paths)
    )
    return sha1(digest_source.encode("utf-8")).hexdigest()[:10]


__all__ = [
    "AblationComparisonArtifacts",
    "AlignedPredictionData",
    "PredictionExport",
    "align_prediction_exports",
    "evaluate_checkpoints_with_eval_config",
    "load_prediction_export",
    "run_ablation_comparison",
]
