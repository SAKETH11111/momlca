"""Tests for ablation comparison CLI helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import compare_ablations
from scripts.compare_ablations import _parse_named_paths


def test_load_confidence_intervals_rejects_non_object_payload(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps([{"aggregate_stats": {}}]))

    with pytest.raises(ValueError, match="must be a JSON object"):
        compare_ablations._load_confidence_intervals({"ModelA": summary_path})


def test_load_confidence_intervals_accepts_mapping_payload(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "aggregate_stats": {
                    "mae_mean": {
                        "n": 3,
                        "mean": 0.2,
                        "ci_half_width": 0.01,
                    }
                }
            }
        )
    )

    loaded = compare_ablations._load_confidence_intervals({"ModelA": summary_path})
    assert loaded["ModelA"]["mae_mean"]["n"] == 3


def test_parse_named_paths_rejects_duplicate_names(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="must be unique"):
        _parse_named_paths(
            [
                f"AblationA={first}",
                f"AblationA={second}",
            ],
            label="prediction export",
        )


def test_parse_named_paths_returns_existing_named_paths(tmp_path: Path) -> None:
    export_path = tmp_path / "export.json"
    export_path.write_text("{}", encoding="utf-8")

    parsed = _parse_named_paths([f"AblationA={export_path}"], label="prediction export")

    assert parsed == {"AblationA": export_path}
