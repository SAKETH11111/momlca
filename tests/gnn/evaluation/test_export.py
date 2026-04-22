"""Tests for evaluation prediction export helpers."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from gnn.evaluation.export import build_prediction_records, export_prediction_records


def test_build_prediction_records_includes_metadata_targets_and_optional_fields() -> None:
    """Export records should include stable metadata and optional uncertainty fields."""
    batches = [
        {
            "predictions": torch.tensor([[1.2, 2.3, 3.4]], dtype=torch.float32),
            "targets": torch.tensor([[1.0, 2.0, float("nan")]], dtype=torch.float32),
            "log_variance": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
            "smiles": ["C(=O)(O)C(F)(F)F"],
            "name": ["TFA"],
            "inchikey": ["ZKHQWZAMYRWXGA-UHFFFAOYSA-N"],
        }
    ]

    records = build_prediction_records(
        prediction_batches=batches,
        split_name="test",
        checkpoint_path="/tmp/checkpoints/best.ckpt",
        property_names=["logS", "logP", "pKa"],
    )

    assert len(records) == 1
    record = records[0]
    assert record["split"] == "test"
    assert record["checkpoint_path"] == "/tmp/checkpoints/best.ckpt"
    assert record["smiles"] == "C(=O)(O)C(F)(F)F"
    assert record["name"] == "TFA"
    assert record["inchikey"] == "ZKHQWZAMYRWXGA-UHFFFAOYSA-N"
    assert record["targets"] == {"logS": 1.0, "logP": 2.0, "pKa": None}
    assert record["predictions"] == {"logS": 1.2, "logP": 2.3, "pKa": 3.4}
    assert record["residuals"] == {"logS": 0.2, "logP": 0.3, "pKa": None}
    assert record["uncertainty"] == {"logS": 0.1, "logP": 0.2, "pKa": 0.3}


def test_export_prediction_records_writes_deterministic_payload(tmp_path: Path) -> None:
    """JSON export should include top-level metadata and records."""
    records = [
        {
            "split": "test",
            "checkpoint_path": "/tmp/checkpoints/best.ckpt",
            "smiles": "C",
            "name": "Methane",
            "inchikey": "VNWKTOKETHGBQD-UHFFFAOYSA-N",
            "targets": {"logS": 0.1},
            "predictions": {"logS": 0.2},
            "residuals": {"logS": 0.1},
        }
    ]

    export_path = export_prediction_records(
        records=records,
        output_dir=tmp_path,
        split_name="test",
        checkpoint_path="/tmp/checkpoints/best.ckpt",
        property_names=["logS"],
    )

    payload = json.loads(export_path.read_text())
    assert payload["metadata"]["split"] == "test"
    assert payload["metadata"]["checkpoint_path"] == "/tmp/checkpoints/best.ckpt"
    assert payload["metadata"]["property_names"] == ["logS"]
    assert payload["metadata"]["num_records"] == 1
    assert payload["records"] == records
