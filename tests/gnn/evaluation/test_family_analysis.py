"""Tests for PFAS family error analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from gnn.evaluation.family_analysis import (
    annotate_family_records,
    load_family_analysis_input,
    run_family_error_analysis,
)


def _write_export(
    path: Path,
    *,
    split: str,
    checkpoint_path: str,
    property_names: list[str],
    records: list[dict[str, object]],
) -> Path:
    payload = {
        "metadata": {
            "split": split,
            "checkpoint_path": checkpoint_path,
            "property_names": property_names,
            "num_records": len(records),
        },
        "records": records,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def _sample_records() -> list[dict[str, object]]:
    return [
        {
            "split": "test",
            "checkpoint_path": "/tmp/models/family.ckpt",
            "smiles": "C(=O)(C(F)(F)F)O",
            "name": "TFA",
            "inchikey": "DTZQGRPZKCHYJP-UHFFFAOYSA-N",
            "targets": {"logS": 1.0, "logP": 0.5},
            "predictions": {"logS": 0.8, "logP": 0.7},
            "residuals": {"logS": -0.2, "logP": 0.2},
        },
        {
            "split": "test",
            "checkpoint_path": "/tmp/models/family.ckpt",
            "smiles": "C(C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)(F)F",
            "name": "PFOS",
            "inchikey": "OSXMWYAPBOPWKO-UHFFFAOYSA-N",
            "targets": {"logS": 2.0, "logP": 1.5},
            "predictions": {"logS": None, "logP": 2.1},
            "residuals": {"logS": None, "logP": 0.6},
        },
        {
            "split": "test",
            "checkpoint_path": "/tmp/models/family.ckpt",
            "smiles": "C(F)(F)(F)C(F)(F)F",
            "name": "Hexafluoroethane",
            "inchikey": "NPVREFRBQCLLIM-UHFFFAOYSA-N",
            "targets": {"logS": 0.2, "logP": 1.2},
            "predictions": {"logS": float("inf"), "logP": 1.0},
            "residuals": {"logS": float("inf"), "logP": -0.2},
        },
    ]


def test_run_family_error_analysis_writes_tables_report_and_figures(tmp_path: Path) -> None:
    export_path = _write_export(
        tmp_path / "family-export.json",
        split="test",
        checkpoint_path="/tmp/models/family.ckpt",
        property_names=["logS", "logP"],
        records=_sample_records(),
    )
    analysis_input = load_family_analysis_input(export_path)
    annotations = annotate_family_records(analysis_input)

    assert len(annotations) == 3
    assert {entry.chain_length for entry in annotations} >= {"C2", "C8"}
    assert {entry.headgroup for entry in annotations} >= {"carboxylate", "sulfonate", "other"}

    artifacts = run_family_error_analysis(
        analysis_input=analysis_input,
        output_dir=tmp_path / "analysis",
        low_sample_threshold=2,
    )

    assert artifacts.chain_length_csv.exists()
    assert artifacts.headgroup_csv.exists()
    assert artifacts.chain_length_figure.exists()
    assert artifacts.headgroup_figure.exists()
    assert artifacts.report_md.exists()

    chain_df = pd.read_csv(artifacts.chain_length_csv)
    head_df = pd.read_csv(artifacts.headgroup_csv)

    required_columns = {
        "family_dimension",
        "family_label",
        "property_name",
        "sample_count",
        "mae",
        "rmse",
        "r2",
        "spearman",
        "mean_signed_residual",
        "mean_absolute_residual",
        "split",
        "checkpoint_path",
        "checkpoint_id",
        "export_id",
    }
    assert required_columns.issubset(set(chain_df.columns))
    assert required_columns.issubset(set(head_df.columns))

    repeated = run_family_error_analysis(
        analysis_input=analysis_input,
        output_dir=tmp_path / "analysis",
        low_sample_threshold=2,
    )
    assert repeated.chain_length_csv.name == artifacts.chain_length_csv.name
    assert repeated.headgroup_figure.name == artifacts.headgroup_figure.name

    report_text = artifacts.report_md.read_text()
    assert "Low-sample families" in report_text
    assert "Worst-performing families" in report_text
    assert "Directional bias highlights" in report_text


def test_family_metrics_mask_missing_and_non_finite_values(tmp_path: Path) -> None:
    records = [
        {
            "split": "test",
            "checkpoint_path": "/tmp/models/family.ckpt",
            "smiles": "C(=O)(C(F)(F)F)O",
            "name": "TFA-1",
            "inchikey": "DTZQGRPZKCHYJP-UHFFFAOYSA-N",
            "targets": {"logS": 1.0},
            "predictions": {"logS": 0.8},
            "residuals": {"logS": -0.2},
        },
        {
            "split": "test",
            "checkpoint_path": "/tmp/models/family.ckpt",
            "smiles": "C(=O)(C(F)(F)F)O",
            "name": "TFA-2",
            "inchikey": "DTZQGRPZKCHYJP-UHFFFAOYSA-N-2",
            "targets": {"logS": 0.9},
            "predictions": {"logS": None},
            "residuals": {"logS": None},
        },
        {
            "split": "test",
            "checkpoint_path": "/tmp/models/family.ckpt",
            "smiles": "C(=O)(C(F)(F)F)O",
            "name": "TFA-3",
            "inchikey": "DTZQGRPZKCHYJP-UHFFFAOYSA-N-3",
            "targets": {"logS": 1.1},
            "predictions": {"logS": float("inf")},
            "residuals": {"logS": float("inf")},
        },
    ]
    export_path = _write_export(
        tmp_path / "masked-export.json",
        split="test",
        checkpoint_path="/tmp/models/family.ckpt",
        property_names=["logS"],
        records=records,
    )
    artifacts = run_family_error_analysis(
        analysis_input=load_family_analysis_input(export_path),
        output_dir=tmp_path / "analysis",
        low_sample_threshold=2,
    )
    chain_df = pd.read_csv(artifacts.chain_length_csv)
    row = chain_df[(chain_df["family_label"] == "C2") & (chain_df["property_name"] == "logS")].iloc[
        0
    ]
    assert int(row["sample_count"]) == 1
    assert float(row["mae"]) == pytest.approx(0.2)
    assert float(row["mean_signed_residual"]) == pytest.approx(-0.2)


def test_load_family_analysis_input_fails_on_split_mixture(tmp_path: Path) -> None:
    records = _sample_records()
    records[1]["split"] = "validation"
    export_path = _write_export(
        tmp_path / "bad-split.json",
        split="test",
        checkpoint_path="/tmp/models/family.ckpt",
        property_names=["logS", "logP"],
        records=records,
    )
    with pytest.raises(ValueError, match="split"):
        run_family_error_analysis(
            analysis_input=load_family_analysis_input(export_path),
            output_dir=tmp_path / "analysis",
        )


def test_load_family_analysis_input_fails_on_mixed_checkpoint_paths(tmp_path: Path) -> None:
    records = _sample_records()
    records[1]["checkpoint_path"] = "/tmp/models/other.ckpt"
    export_path = _write_export(
        tmp_path / "mixed-checkpoints.json",
        split="test",
        checkpoint_path="/tmp/models/family.ckpt",
        property_names=["logS", "logP"],
        records=records,
    )
    with pytest.raises(ValueError, match="mixes record checkpoint_path values"):
        load_family_analysis_input(export_path)


def test_annotation_fails_on_missing_smiles(tmp_path: Path) -> None:
    records = _sample_records()
    records[0]["smiles"] = ""
    export_path = _write_export(
        tmp_path / "missing-smiles.json",
        split="test",
        checkpoint_path="/tmp/models/family.ckpt",
        property_names=["logS", "logP"],
        records=records,
    )
    with pytest.raises(ValueError, match="SMILES"):
        annotate_family_records(load_family_analysis_input(export_path))


def test_annotation_fails_on_duplicate_sample_key(tmp_path: Path) -> None:
    records = _sample_records()
    records[1]["smiles"] = records[0]["smiles"]
    records[1]["name"] = records[0]["name"]
    records[1]["inchikey"] = records[0]["inchikey"]
    export_path = _write_export(
        tmp_path / "duplicate.json",
        split="test",
        checkpoint_path="/tmp/models/family.ckpt",
        property_names=["logS", "logP"],
        records=records,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        annotate_family_records(load_family_analysis_input(export_path))


def test_annotation_fails_on_property_mismatch(tmp_path: Path) -> None:
    records = _sample_records()
    records[0]["predictions"] = {"logS": 0.8}
    export_path = _write_export(
        tmp_path / "property-mismatch.json",
        split="test",
        checkpoint_path="/tmp/models/family.ckpt",
        property_names=["logS", "logP"],
        records=records,
    )
    with pytest.raises(ValueError, match="property"):
        annotate_family_records(load_family_analysis_input(export_path))
