"""Tests for ablation comparison CLI helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compare_ablations import _parse_named_paths


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
