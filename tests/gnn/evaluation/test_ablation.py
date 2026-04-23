"""Tests for checkpoint ablation comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gnn.evaluation.ablation import (
    align_prediction_exports,
    load_prediction_export,
    run_ablation_comparison,
)
from gnn.evaluation.significance import (
    build_pairwise_significance_table,
    run_paired_significance_test,
)


def _write_export(
    path: Path,
    *,
    split: str,
    checkpoint_path: str,
    records: list[dict[str, object]],
    property_names: list[str],
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


def _sample_records(
    *,
    log_s_predictions: tuple[float, float],
) -> list[dict[str, object]]:
    return [
        {
            "split": "test",
            "checkpoint_path": "/tmp/checkpoints/a.ckpt",
            "smiles": "CCO",
            "name": "ethanol",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "targets": {"logS": 1.0, "logP": 2.0},
            "predictions": {"logS": log_s_predictions[0], "logP": 2.1},
            "residuals": {"logS": log_s_predictions[0] - 1.0, "logP": 0.1},
        },
        {
            "split": "test",
            "checkpoint_path": "/tmp/checkpoints/a.ckpt",
            "smiles": "CCN",
            "name": "ethylamine",
            "inchikey": "QUSNBJAOOMFDIB-UHFFFAOYSA-N",
            "targets": {"logS": 2.0, "logP": 1.0},
            "predictions": {"logS": log_s_predictions[1], "logP": 1.2},
            "residuals": {"logS": log_s_predictions[1] - 2.0, "logP": 0.2},
        },
    ]


def test_align_prediction_exports_builds_aligned_target_and_prediction_matrices(
    tmp_path: Path,
) -> None:
    export_a = _write_export(
        tmp_path / "a.json",
        split="test",
        checkpoint_path="/tmp/models/ablation/model.ckpt",
        property_names=["logS", "logP"],
        records=_sample_records(log_s_predictions=(1.0, 2.1)),
    )
    export_b = _write_export(
        tmp_path / "b.json",
        split="test",
        checkpoint_path="/tmp/models/ablation2/model.ckpt",
        property_names=["logS", "logP"],
        records=_sample_records(log_s_predictions=(1.3, 2.4)),
    )

    aligned = align_prediction_exports(
        {
            "AblationA": load_prediction_export(export_a, model_name="AblationA"),
            "AblationB": load_prediction_export(export_b, model_name="AblationB"),
        }
    )

    assert aligned.split_name == "test"
    assert aligned.property_names == ["logS", "logP"]
    assert aligned.y_true.shape == (2, 2)
    assert np.allclose(aligned.predictions_by_model["AblationA"][:, 0], [1.0, 2.1])
    assert np.allclose(aligned.predictions_by_model["AblationB"][:, 0], [1.3, 2.4])


def test_align_prediction_exports_fails_on_split_mismatch(tmp_path: Path) -> None:
    export_a = _write_export(
        tmp_path / "a.json",
        split="test",
        checkpoint_path="/tmp/models/a.ckpt",
        property_names=["logS"],
        records=_sample_records(log_s_predictions=(1.0, 2.1)),
    )
    export_b = _write_export(
        tmp_path / "b.json",
        split="validation",
        checkpoint_path="/tmp/models/b.ckpt",
        property_names=["logS"],
        records=_sample_records(log_s_predictions=(1.2, 2.2)),
    )

    with pytest.raises(ValueError, match="split"):
        align_prediction_exports(
            {
                "A": load_prediction_export(export_a, model_name="A"),
                "B": load_prediction_export(export_b, model_name="B"),
            }
        )


def test_align_prediction_exports_fails_on_missing_rows(tmp_path: Path) -> None:
    records = _sample_records(log_s_predictions=(1.0, 2.1))
    export_a = _write_export(
        tmp_path / "a.json",
        split="test",
        checkpoint_path="/tmp/models/a.ckpt",
        property_names=["logS", "logP"],
        records=records,
    )
    export_b = _write_export(
        tmp_path / "b.json",
        split="test",
        checkpoint_path="/tmp/models/b.ckpt",
        property_names=["logS", "logP"],
        records=records[:1],
    )

    with pytest.raises(ValueError, match="aligned"):
        align_prediction_exports(
            {
                "A": load_prediction_export(export_a, model_name="A"),
                "B": load_prediction_export(export_b, model_name="B"),
            }
        )


def test_align_prediction_exports_uses_stable_identifiers_not_display_name(tmp_path: Path) -> None:
    records_a = _sample_records(log_s_predictions=(1.0, 2.1))
    records_b = _sample_records(log_s_predictions=(1.2, 2.3))
    records_b[0]["name"] = "ethanol (renamed)"
    records_b[1]["name"] = "ethylamine (renamed)"

    export_a = _write_export(
        tmp_path / "a.json",
        split="test",
        checkpoint_path="/tmp/models/a.ckpt",
        property_names=["logS", "logP"],
        records=records_a,
    )
    export_b = _write_export(
        tmp_path / "b.json",
        split="test",
        checkpoint_path="/tmp/models/b.ckpt",
        property_names=["logS", "logP"],
        records=records_b,
    )

    aligned = align_prediction_exports(
        {
            "A": load_prediction_export(export_a, model_name="A"),
            "B": load_prediction_export(export_b, model_name="B"),
        }
    )

    assert len(aligned.sample_keys) == 2
    assert all(len(key) == 2 for key in aligned.sample_keys)


def test_align_prediction_exports_fails_on_duplicate_identifiers_with_different_names(
    tmp_path: Path,
) -> None:
    duplicate_records = _sample_records(log_s_predictions=(1.0, 2.1))
    duplicate_records.append(
        {
            "split": "test",
            "checkpoint_path": "/tmp/checkpoints/a.ckpt",
            "smiles": "CCO",
            "name": "ethanol duplicate label",
            "inchikey": "LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
            "targets": {"logS": 1.0, "logP": 2.0},
            "predictions": {"logS": 1.0, "logP": 2.1},
            "residuals": {"logS": 0.0, "logP": 0.1},
        }
    )
    export_path = _write_export(
        tmp_path / "dup.json",
        split="test",
        checkpoint_path="/tmp/models/a.ckpt",
        property_names=["logS", "logP"],
        records=duplicate_records,
    )
    export = load_prediction_export(export_path, model_name="A")

    with pytest.raises(ValueError, match="Duplicate sample key"):
        align_prediction_exports({"A": export, "B": export})


def test_build_pairwise_significance_table_reports_winner_direction() -> None:
    y_true = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=float)
    predictions_by_model = {
        "A": np.array([[1.0], [2.0], [3.1], [3.9]], dtype=float),
        "B": np.array([[1.8], [2.7], [2.2], [4.9]], dtype=float),
    }

    significance = build_pairwise_significance_table(
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        property_names=["logS"],
    )

    assert len(significance) == 1
    row = significance.iloc[0]
    assert row["metric_proxy"] == "absolute_error"
    assert int(row["sample_count"]) == 4
    assert row["test_name"] == "wilcoxon"
    assert row["winning_direction"].startswith("A")


def test_build_pairwise_significance_table_supports_ttest_rel() -> None:
    y_true = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
    predictions_by_model = {
        "A": np.array([[0.0], [1.1], [2.1], [3.0]], dtype=float),
        "B": np.array([[0.2], [0.8], [2.6], [2.4]], dtype=float),
    }

    significance = build_pairwise_significance_table(
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        property_names=["logS"],
        test_name="ttest_rel",
    )

    assert len(significance) == 1
    row = significance.iloc[0]
    assert row["test_name"] == "ttest_rel"
    assert np.isfinite(float(row["statistic"]))
    assert np.isfinite(float(row["p_value"]))


def test_run_paired_significance_test_ttest_rel_identical_arrays_returns_non_significant() -> None:
    identical_errors = np.array([1.0, 2.0, 3.0], dtype=float)

    result = run_paired_significance_test(
        identical_errors,
        identical_errors.copy(),
        test_name="ttest_rel",
    )

    assert result.sample_count == 3
    assert result.test_name == "ttest_rel"
    assert result.statistic == 0.0
    assert result.p_value == 1.0


@pytest.mark.parametrize("test_name", ["ttest_rel", "wilcoxon"])
def test_run_paired_significance_test_zero_samples_returns_insufficient_result(
    test_name: str,
) -> None:
    first_errors = np.array([np.nan, np.nan], dtype=float)
    second_errors = np.array([np.nan, np.nan], dtype=float)

    result = run_paired_significance_test(first_errors, second_errors, test_name=test_name)

    assert result.sample_count == 0
    assert result.test_name == f"{test_name}(insufficient_samples)"
    assert np.isnan(result.statistic)
    assert np.isnan(result.p_value)


@pytest.mark.parametrize("test_name", ["ttest_rel", "wilcoxon"])
def test_run_paired_significance_test_near_equal_arrays_do_not_short_circuit(
    test_name: str,
) -> None:
    first_errors = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    second_errors = np.array([1.000005, 1.999996, 3.000003, 3.999998], dtype=float)
    assert np.allclose(first_errors, second_errors)
    assert not np.array_equal(first_errors, second_errors)

    result = run_paired_significance_test(first_errors, second_errors, test_name=test_name)

    assert result.sample_count == 4
    assert result.test_name == test_name
    assert (result.statistic, result.p_value) != (0.0, 1.0)


def test_build_pairwise_significance_table_handles_fully_masked_columns() -> None:
    y_true = np.array([[np.nan], [np.nan]], dtype=float)
    predictions_by_model = {
        "A": np.array([[1.0], [2.0]], dtype=float),
        "B": np.array([[1.5], [2.5]], dtype=float),
    }

    significance = build_pairwise_significance_table(
        y_true=y_true,
        predictions_by_model=predictions_by_model,
        property_names=["logS"],
    )

    assert len(significance) == 1
    row = significance.iloc[0]
    assert int(row["sample_count"]) == 0
    assert row["test_name"] == "wilcoxon(insufficient_samples)"
    assert np.isnan(float(row["p_value"]))


def test_run_ablation_comparison_writes_csv_and_report_artifacts(tmp_path: Path) -> None:
    export_a = _write_export(
        tmp_path / "a.json",
        split="test",
        checkpoint_path="/tmp/exp1/model.ckpt",
        property_names=["logS"],
        records=_sample_records(log_s_predictions=(1.0, 2.0)),
    )
    export_b = _write_export(
        tmp_path / "b.json",
        split="test",
        checkpoint_path="/tmp/exp2/model.ckpt",
        property_names=["logS"],
        records=_sample_records(log_s_predictions=(1.4, 2.7)),
    )

    artifacts = run_ablation_comparison(
        {
            "A": load_prediction_export(export_a, model_name="A"),
            "B": load_prediction_export(export_b, model_name="B"),
        },
        output_dir=tmp_path / "reports",
    )

    assert artifacts.comparison_csv.exists()
    assert artifacts.significance_csv.exists()
    assert artifacts.report_md.exists()

    comparison_df = pd.read_csv(artifacts.comparison_csv, index_col=0)
    assert "checkpoint_path" in comparison_df.columns
    assert "checkpoint_id" in comparison_df.columns


def test_run_ablation_comparison_passes_confidence_intervals_to_reports(tmp_path: Path) -> None:
    export_a = _write_export(
        tmp_path / "a.json",
        split="test",
        checkpoint_path="/tmp/exp1/model.ckpt",
        property_names=["logS"],
        records=_sample_records(log_s_predictions=(1.0, 2.0)),
    )
    export_b = _write_export(
        tmp_path / "b.json",
        split="test",
        checkpoint_path="/tmp/exp2/model.ckpt",
        property_names=["logS"],
        records=_sample_records(log_s_predictions=(1.4, 2.7)),
    )

    artifacts = run_ablation_comparison(
        {
            "A": load_prediction_export(export_a, model_name="A"),
            "B": load_prediction_export(export_b, model_name="B"),
        },
        output_dir=tmp_path / "reports",
        confidence_intervals_by_model={
            "A": {
                "mae_mean": {
                    "n": 3,
                    "mean": 0.25,
                    "std": 0.03,
                    "sem": 0.01732,
                    "ci_method": "normal",
                    "ci_level": 0.95,
                    "ci_low": 0.23,
                    "ci_high": 0.27,
                    "ci_half_width": 0.02,
                    "ci95": 0.02,
                }
            }
        },
    )

    comparison_df = pd.read_csv(artifacts.comparison_csv, index_col=0)
    assert "mae_mean_ci_method" in comparison_df.columns
    assert "mae_mean_ci_display" in comparison_df.columns
    assert comparison_df.loc["A", "mae_mean_ci_method"] == "normal"
    assert comparison_df.loc["A", "mae_mean_ci_display"] == "0.2500 +/- 0.0200"
