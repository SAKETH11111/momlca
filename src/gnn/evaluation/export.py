"""Prediction export helpers for evaluation-time JSON artifacts."""

from __future__ import annotations

import json
import logging
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from lightning.pytorch.loggers import Logger
from lightning_utilities.core.rank_zero import rank_zero_only

logger = logging.getLogger(__name__)


def _as_float(value: float) -> float | None:
    scalar = float(value)
    if not math.isfinite(scalar):
        return None
    return round(scalar, 6)


def _ensure_tensor(
    value: Any,
    *,
    key: str,
    expected_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Prediction batch field '{key}' must be a torch.Tensor")

    tensor = value.detach().cpu()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim != 2:
        raise ValueError(
            f"Prediction batch field '{key}' must be 2D after normalization; got {tensor.shape}"
        )
    if expected_shape is not None and tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"Prediction batch field '{key}' shape {tuple(tensor.shape)} "
            f"does not match expected shape {expected_shape}"
        )
    return tensor


def _normalize_text_list(value: Any, *, key: str, size: int) -> list[str]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        normalized = [str(item) for item in value]
    elif value is None:
        normalized = ["" for _ in range(size)]
    else:
        normalized = [str(value) for _ in range(size)]

    if len(normalized) != size:
        raise ValueError(
            f"Prediction batch field '{key}' must have {size} entries, got {len(normalized)}"
        )
    return normalized


def _resolve_property_names(
    batch_value: Any,
    fallback_names: Sequence[str] | None,
    *,
    num_targets: int,
) -> list[str]:
    if isinstance(batch_value, Sequence) and not isinstance(batch_value, (str, bytes)):
        names = [str(name) for name in batch_value]
    elif fallback_names is not None:
        names = [str(name) for name in fallback_names]
    else:
        names = [f"target_{index}" for index in range(num_targets)]

    if len(names) != num_targets:
        raise ValueError(
            f"Property names must match prediction width ({num_targets}), got {len(names)}"
        )
    return names


def _tensor_row_to_mapping(values: torch.Tensor, names: Sequence[str]) -> dict[str, float | None]:
    return {name: _as_float(float(values[index])) for index, name in enumerate(names)}


def build_prediction_records(
    *,
    prediction_batches: Sequence[Mapping[str, Any]],
    split_name: str,
    checkpoint_path: str,
    property_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Convert model prediction batches into a stable JSON-record schema."""
    records: list[dict[str, Any]] = []
    for batch in prediction_batches:
        predictions = _ensure_tensor(batch.get("predictions"), key="predictions")
        num_rows, num_targets = predictions.shape
        names = _resolve_property_names(
            batch.get("property_names"),
            property_names,
            num_targets=num_targets,
        )

        targets = batch.get("targets")
        target_tensor = (
            _ensure_tensor(targets, key="targets", expected_shape=(num_rows, num_targets))
            if targets is not None
            else None
        )
        log_variance = batch.get("log_variance")
        uncertainty_tensor = (
            _ensure_tensor(
                log_variance,
                key="log_variance",
                expected_shape=(num_rows, num_targets),
            )
            if log_variance is not None
            else None
        )

        smiles = _normalize_text_list(batch.get("smiles"), key="smiles", size=num_rows)
        names_list = _normalize_text_list(batch.get("name"), key="name", size=num_rows)
        inchikeys = _normalize_text_list(batch.get("inchikey"), key="inchikey", size=num_rows)

        for row_index in range(num_rows):
            prediction_map = _tensor_row_to_mapping(predictions[row_index], names)
            target_map = (
                _tensor_row_to_mapping(target_tensor[row_index], names)
                if target_tensor is not None
                else dict.fromkeys(names)
            )
            residual_map: dict[str, float | None] = dict.fromkeys(names)
            if target_tensor is not None:
                for property_index, name in enumerate(names):
                    prediction_value = prediction_map[name]
                    target_value = target_map[name]
                    residual_map[name] = (
                        None
                        if prediction_value is None or target_value is None
                        else _as_float(
                            float(predictions[row_index, property_index])
                            - float(target_tensor[row_index, property_index])
                        )
                    )

            record: dict[str, Any] = {
                "split": split_name,
                "checkpoint_path": checkpoint_path,
                "smiles": smiles[row_index],
                "name": names_list[row_index],
                "inchikey": inchikeys[row_index],
                "targets": target_map,
                "predictions": prediction_map,
                "residuals": residual_map,
            }
            if uncertainty_tensor is not None:
                record["uncertainty"] = _tensor_row_to_mapping(
                    uncertainty_tensor[row_index],
                    names,
                )
            records.append(record)

    return records


def export_prediction_records(
    *,
    records: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    split_name: str,
    checkpoint_path: str,
    property_names: Sequence[str] | None,
) -> Path:
    """Write prediction records to a deterministic JSON artifact path."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    ckpt_stem = Path(str(checkpoint_path)).stem or "checkpoint"
    safe_ckpt_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", ckpt_stem).strip("-")
    if safe_ckpt_stem == "":
        safe_ckpt_stem = "checkpoint"
    export_path = output_dir_path / f"{split_name}-{safe_ckpt_stem}-predictions.json"

    payload = {
        "metadata": {
            "split": split_name,
            "checkpoint_path": checkpoint_path,
            "property_names": list(property_names) if property_names is not None else [],
            "num_records": len(records),
        },
        "records": list(records),
    }
    export_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return export_path


@rank_zero_only
def maybe_log_prediction_artifact(
    *,
    loggers: Sequence[Logger],
    prediction_path: Path,
    records: Sequence[Mapping[str, Any]],
    artifact_name: str,
    split_name: str,
    max_rows: int = 25,
) -> None:
    """Log prediction export artifacts/tables via WandbLogger when available."""
    try:
        import wandb
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        return

    for logger_instance in loggers:
        if not isinstance(logger_instance, WandbLogger):
            continue
        run = getattr(logger_instance, "experiment", None)
        if run is None:
            continue

        try:
            artifact = wandb.Artifact(name=f"{artifact_name}-{split_name}", type="prediction")
            artifact.add_file(str(prediction_path), name=prediction_path.name)
            run.log_artifact(artifact)

            if max_rows > 0 and records:
                preview_rows = list(records[:max_rows])
                table = wandb.Table(
                    columns=[
                        "split",
                        "smiles",
                        "name",
                        "inchikey",
                        "targets",
                        "predictions",
                    ],
                    data=[
                        [
                            row.get("split", ""),
                            row.get("smiles", ""),
                            row.get("name", ""),
                            row.get("inchikey", ""),
                            json.dumps(row.get("targets", {}), sort_keys=True),
                            json.dumps(row.get("predictions", {}), sort_keys=True),
                        ]
                        for row in preview_rows
                    ],
                )
                run.log({f"{artifact_name}/{split_name}_preview": table})
        except Exception:
            logger.exception("Failed to log prediction artifact/table to Weights & Biases.")
