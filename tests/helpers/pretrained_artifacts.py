"""Helpers for locating DVC-tracked pretrained artifacts in tests."""

from __future__ import annotations

from pathlib import Path

TRACKED_PAINN_STAGE_ARTIFACT_RELATIVE_PATH = (
    "artifacts/pretrained/isdpainn_random_split_painn_stage_backbone.pt"
)


def get_tracked_painn_stage_artifact_path() -> Path:
    """Return the absolute path to the tracked PaiNN-stage pretrained checkpoint."""
    return Path(__file__).resolve().parents[2] / TRACKED_PAINN_STAGE_ARTIFACT_RELATIVE_PATH


def require_tracked_painn_stage_artifact() -> Path:
    """Return the tracked checkpoint path or fail if DVC data is not materialized."""
    artifact_path = get_tracked_painn_stage_artifact_path()
    if not artifact_path.exists():
        raise FileNotFoundError(
            "Tracked pretrained artifact is not materialized. Run `poetry run dvc pull` before pytest."
        )
    return artifact_path
