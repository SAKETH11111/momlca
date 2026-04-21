"""Regression tests for the pretrained-backbone preparation workflow."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

import scripts.prepare_pretrained_backbone as prepare_pretrained_backbone


def test_default_upstream_cache_uses_system_temp_dir() -> None:
    """Default upstream cache paths should stay outside the repo worktree."""
    temp_dir = Path(tempfile.gettempdir()).resolve()

    assert prepare_pretrained_backbone.DEFAULT_SOURCE_PATH.is_absolute()
    assert prepare_pretrained_backbone.DEFAULT_CONFIG_PATH.is_absolute()
    assert prepare_pretrained_backbone.DEFAULT_SOURCE_PATH.is_relative_to(temp_dir)
    assert prepare_pretrained_backbone.DEFAULT_CONFIG_PATH.is_relative_to(temp_dir)
    assert not prepare_pretrained_backbone.DEFAULT_SOURCE_PATH.is_relative_to(
        prepare_pretrained_backbone.PROJECT_ROOT
    )
    assert not prepare_pretrained_backbone.DEFAULT_CONFIG_PATH.is_relative_to(
        prepare_pretrained_backbone.PROJECT_ROOT
    )


def test_ensure_download_validates_pinned_sha256_before_accepting_file(
    tmp_path: Path,
) -> None:
    """Pinned SHA-256 digests should be enforced for already-materialized upstream files."""
    source_path = tmp_path / "upstream.bin"
    source_path.write_bytes(b"tracked-upstream-weights")

    prepare_pretrained_backbone.ensure_download(
        path=source_path,
        url="https://example.invalid/upstream.bin",
        expected_digests={
            "sha256": prepare_pretrained_backbone.compute_digest(source_path, algorithm="sha256"),
            "md5": prepare_pretrained_backbone.compute_digest(source_path, algorithm="md5"),
        },
        skip_download=True,
    )

    with pytest.raises(ValueError, match="sha256"):
        prepare_pretrained_backbone.ensure_download(
            path=source_path,
            url="https://example.invalid/upstream.bin",
            expected_digests={
                "sha256": "0" * 64,
                "md5": prepare_pretrained_backbone.compute_digest(source_path, algorithm="md5"),
            },
            skip_download=True,
        )


def test_prepare_artifact_uses_safe_load_and_records_upstream_retrieval_dates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation should use weights-only loading and persist upstream retrieval timestamps."""
    source_checkpoint_path = tmp_path / "model_state_random_split.pt"
    source_config_path = tmp_path / "config-defaults_random_split.yaml"
    output_path = tmp_path / "artifacts" / "prepared.pt"

    torch.save(
        {
            "model_state_dict": {
                "invariant_atom_emb.embeddings.weight": torch.arange(
                    9 * 512, dtype=torch.float32
                ).reshape(9, 512)
            }
        },
        source_checkpoint_path,
    )
    source_config_path.write_text("model:\n  hidden_channels: 512\n", encoding="utf-8")

    checkpoint_timestamp = 1_710_000_000
    config_timestamp = 1_710_000_600
    os.utime(source_checkpoint_path, (checkpoint_timestamp, checkpoint_timestamp))
    os.utime(source_config_path, (config_timestamp, config_timestamp))

    monkeypatch.setattr(
        prepare_pretrained_backbone,
        "UPSTREAM_SOURCE_SHA256",
        prepare_pretrained_backbone.compute_digest(source_checkpoint_path, algorithm="sha256"),
    )
    monkeypatch.setattr(
        prepare_pretrained_backbone,
        "UPSTREAM_SOURCE_MD5",
        prepare_pretrained_backbone.compute_digest(source_checkpoint_path, algorithm="md5"),
    )
    monkeypatch.setattr(
        prepare_pretrained_backbone,
        "UPSTREAM_CONFIG_SHA256",
        prepare_pretrained_backbone.compute_digest(source_config_path, algorithm="sha256"),
    )
    monkeypatch.setattr(
        prepare_pretrained_backbone,
        "UPSTREAM_CONFIG_MD5",
        prepare_pretrained_backbone.compute_digest(source_config_path, algorithm="md5"),
    )

    original_torch_load = prepare_pretrained_backbone.torch.load
    captured_load_kwargs: dict[str, object] = {}

    def recording_torch_load(*args: object, **kwargs: object) -> object:
        captured_load_kwargs.update(kwargs)
        return original_torch_load(*args, **kwargs)

    monkeypatch.setattr(prepare_pretrained_backbone.torch, "load", recording_torch_load)

    prepared = prepare_pretrained_backbone.prepare_artifact(
        source_checkpoint_path=source_checkpoint_path,
        source_config_path=source_config_path,
        output_path=output_path,
        metadata_output_path=None,
        skip_download=True,
    )

    metadata = yaml.safe_load(prepared.metadata_path.read_text(encoding="utf-8"))
    snapshot = metadata["source"]["upstream_snapshot"]

    assert captured_load_kwargs["weights_only"] is True
    assert captured_load_kwargs["map_location"] == "cpu"
    assert snapshot["checkpoint_retrieved_at"] == "2024-03-09T16:00:00Z"
    assert snapshot["config_retrieved_at"] == "2024-03-09T16:10:00Z"
