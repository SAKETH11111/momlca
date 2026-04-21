"""Prepare a repo-compatible pretrained backbone checkpoint from a public source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

import rootutils
import torch
import yaml

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gnn.data.transforms.constants import ATOM_ATOMIC_NUMBER_SLICE, ATOM_FEATURE_DIM

UPSTREAM_SOURCE_URL = (
    "https://zenodo.org/api/records/10547719/files/model_state_random_split.pt/content"
)
UPSTREAM_CONFIG_URL = (
    "https://zenodo.org/api/records/10547719/files/config-defaults_random_split.yaml/content"
)
UPSTREAM_SOURCE_SHA256 = "9473a3d62485297825e43dab47d9fd6be9956487e212c9143e1663f5538cf3ec"
UPSTREAM_SOURCE_MD5 = "5265a7d187000628c2b71bc26e9bd46f"
UPSTREAM_CONFIG_SHA256 = "0cac15db2931da1cdf86fea638a22d7507b160db4a14e8dc6c899542a36f186f"
UPSTREAM_CONFIG_MD5 = "de5c3250ca57531e7a43acdd49687ddf"
UPSTREAM_ZENODO_DOI = "10.5281/zenodo.10547719"
UPSTREAM_ZENODO_RECORD_ID = 10547719
UPSTREAM_REPO_URL = "https://github.com/nmdl-mizo/isdpainn"
UPSTREAM_REPO_TAG = "v1.0.0"
UPSTREAM_REPO_TAG_COMMIT = "a2138f6ecf1a14124fc9c8c6c598bad5f01abd11"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_RELATIVE_PATH = Path("scripts/prepare_pretrained_backbone.py")
CONSTANTS_MODULE_RELATIVE_PATH = Path("src/gnn/data/transforms/constants.py")
CONVERSION_SCHEMA_VERSION = 1

DEFAULT_UPSTREAM_CACHE_DIR = (
    Path(tempfile.gettempdir()).resolve() / "moml" / "prepare_pretrained_backbone"
)
DEFAULT_SOURCE_PATH = DEFAULT_UPSTREAM_CACHE_DIR / "model_state_random_split.pt"
DEFAULT_CONFIG_PATH = DEFAULT_UPSTREAM_CACHE_DIR / "config-defaults_random_split.yaml"
DEFAULT_ARTIFACT_PATH = Path("artifacts/pretrained/isdpainn_random_split_painn_stage_backbone.pt")

UPSTREAM_ATOMIC_NUMBERS = (1, 6, 7, 8, 9)
DIRECT_ELEMENT_COLUMNS = {
    6: 0,  # C
    7: 1,  # N
    8: 2,  # O
    9: 3,  # F
}
FALLBACK_ELEMENT_COLUMNS = (4, 5, 6, 7, 8)  # P, S, Cl, Br, I
OTHER_ELEMENT_COLUMN = 9
TARGET_HIDDEN_CHANNELS = 128


@dataclass(frozen=True)
class PreparedArtifact:
    """Metadata about the prepared checkpoint artifact."""

    artifact_path: Path
    metadata_path: Path
    sha256: str
    size_bytes: int
    source_sha256: str
    config_sha256: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_PATH,
        help="Local path for the downloaded upstream checkpoint.",
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Local path for the downloaded upstream config YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Output path for the repo-compatible checkpoint.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="Optional explicit metadata output path.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Require existing local upstream files instead of downloading them.",
    )
    return parser.parse_args()


def verify_expected_digests(path: Path, expected_digests: dict[str, str]) -> None:
    """Validate all pinned digests for ``path``."""
    for algorithm, expected_digest in expected_digests.items():
        actual_digest = compute_digest(path, algorithm=algorithm)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Digest mismatch for {path}: expected {algorithm}={expected_digest}, "
                f"got {algorithm}={actual_digest}"
            )


def ensure_download(
    path: Path, url: str, expected_digests: dict[str, str], skip_download: bool
) -> None:
    """Download ``url`` to ``path`` when needed and verify its pinned digests."""
    if not path.exists():
        if skip_download:
            raise FileNotFoundError(f"Missing required local source file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urlopen(url) as response, path.open("wb") as output_file:
            output_file.write(response.read())

    verify_expected_digests(path, expected_digests)


def compute_digest(path: Path, algorithm: str) -> str:
    """Compute a hex digest for ``path`` with the requested hash algorithm."""
    hasher = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_repo_head_commit(repo_root: Path) -> str | None:
    """Return the current git HEAD commit when available."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    head_commit = result.stdout.strip()
    return head_commit or None


def build_repo_feature_schema() -> dict[str, object]:
    """Describe the repo-side feature schema that the conversion depends on."""
    return {
        "atom_feature_dim": ATOM_FEATURE_DIM,
        "atom_atomic_number_slice": [
            ATOM_ATOMIC_NUMBER_SLICE.start,
            ATOM_ATOMIC_NUMBER_SLICE.stop,
        ],
        "upstream_atomic_numbers": list(UPSTREAM_ATOMIC_NUMBERS),
        "direct_element_columns": DIRECT_ELEMENT_COLUMNS,
        "fallback_element_columns": list(FALLBACK_ELEMENT_COLUMNS),
        "other_element_column": OTHER_ELEMENT_COLUMN,
        "target_hidden_channels": TARGET_HIDDEN_CHANNELS,
    }


def compute_schema_sha256(schema_inputs: dict[str, object]) -> str:
    """Compute a stable digest for the repo-side conversion schema inputs."""
    payload = json.dumps(schema_inputs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def format_file_timestamp(path: Path) -> str:
    """Return a stable UTC timestamp for when ``path`` was last materialized locally."""
    return (
        datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def reduce_embedding_width(embedding: torch.Tensor, target_width: int) -> torch.Tensor:
    """Reduce an embedding matrix width by mean-pooling fixed contiguous blocks."""
    if embedding.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {tuple(embedding.shape)}")
    if embedding.shape[1] % target_width != 0:
        raise ValueError(
            f"Cannot reduce embedding width {embedding.shape[1]} evenly to {target_width}"
        )

    block_width = embedding.shape[1] // target_width
    return embedding.reshape(embedding.shape[0], target_width, block_width).mean(dim=-1)


def build_repo_backbone_state_dict(
    source_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Convert upstream atom embeddings into the repo's ``PaiNNStageBackbone`` contract."""
    try:
        source_embedding = source_state["invariant_atom_emb.embeddings.weight"]
    except KeyError as error:
        raise KeyError(
            "Upstream checkpoint does not expose 'invariant_atom_emb.embeddings.weight'"
        ) from error

    reduced_embedding = reduce_embedding_width(
        source_embedding, target_width=TARGET_HIDDEN_CHANNELS
    )
    available_embeddings = {
        atomic_number: reduced_embedding[atomic_number - 1].clone()
        for atomic_number in UPSTREAM_ATOMIC_NUMBERS
    }

    fallback_embedding = torch.stack(
        [available_embeddings[atomic_number] for atomic_number in UPSTREAM_ATOMIC_NUMBERS],
        dim=0,
    ).mean(dim=0)
    other_embedding = available_embeddings[1]

    input_dim = ATOM_FEATURE_DIM
    if ATOM_ATOMIC_NUMBER_SLICE.stop != 10:
        raise ValueError(
            "The conversion script assumes a 10-column repo atomic-number one-hot slice."
        )

    weight = torch.zeros((TARGET_HIDDEN_CHANNELS, input_dim), dtype=torch.float32)
    bias = torch.zeros((TARGET_HIDDEN_CHANNELS,), dtype=torch.float32)

    for atomic_number, column_index in DIRECT_ELEMENT_COLUMNS.items():
        weight[:, column_index] = available_embeddings[atomic_number]

    for column_index in FALLBACK_ELEMENT_COLUMNS:
        weight[:, column_index] = fallback_embedding

    weight[:, OTHER_ELEMENT_COLUMN] = other_embedding

    return {
        "backbone.node_projection.weight": weight.contiguous(),
        "backbone.node_projection.bias": bias,
    }


def build_metadata(
    prepared: PreparedArtifact,
    source_checkpoint_path: Path,
    source_config_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Build provenance metadata for the prepared artifact."""
    script_path = PROJECT_ROOT / SCRIPT_RELATIVE_PATH
    constants_module_path = PROJECT_ROOT / CONSTANTS_MODULE_RELATIVE_PATH
    schema_inputs = {
        "schema_version": CONVERSION_SCHEMA_VERSION,
        "repo_head_commit": get_repo_head_commit(PROJECT_ROOT),
        "script": SCRIPT_RELATIVE_PATH.as_posix(),
        "script_sha256": compute_digest(script_path, algorithm="sha256"),
        "constants_module": CONSTANTS_MODULE_RELATIVE_PATH.as_posix(),
        "constants_module_sha256": compute_digest(constants_module_path, algorithm="sha256"),
        "repo_feature_schema": build_repo_feature_schema(),
    }

    return {
        "artifact": {
            "path": output_path.as_posix(),
            "format": "state_dict",
            "checkpoint_format": "state_dict",
            "backbone_key_prefix": "backbone.",
            "compatible_backbone_target": "gnn.models.backbones.PaiNNStageBackbone",
            "generated_at": datetime.now(UTC).date().isoformat(),
            "sha256": prepared.sha256,
            "file_size_bytes": prepared.size_bytes,
        },
        "source": {
            "type": "converted-public-pretrained-checkpoint",
            "upstream_model": "ISD-PaiNN",
            "upstream_repo_url": UPSTREAM_REPO_URL,
            "upstream_repo_tag": UPSTREAM_REPO_TAG,
            "upstream_repo_tag_commit": UPSTREAM_REPO_TAG_COMMIT,
            "upstream_snapshot": {
                "zenodo_doi": UPSTREAM_ZENODO_DOI,
                "zenodo_record_id": UPSTREAM_ZENODO_RECORD_ID,
                "checkpoint_file": source_checkpoint_path.name,
                "checkpoint_url": UPSTREAM_SOURCE_URL,
                "checkpoint_retrieved_at": format_file_timestamp(source_checkpoint_path),
                "checkpoint_md5": UPSTREAM_SOURCE_MD5,
                "checkpoint_sha256": prepared.source_sha256,
                "config_file": source_config_path.name,
                "config_url": UPSTREAM_CONFIG_URL,
                "config_retrieved_at": format_file_timestamp(source_config_path),
                "config_md5": UPSTREAM_CONFIG_MD5,
                "config_sha256": prepared.config_sha256,
            },
            "training_dataset_name": "Simulated carbon K-edge spectra dataset aligned to QM9 molecules",
            "training_dataset_reference": "https://doi.org/10.1038/s41597-022-01303-8",
            "license": {
                "weights": "CC-BY-4.0",
                "code": "MIT",
                "usage_constraints": "Retain attribution for the Zenodo weights and upstream ISD-PaiNN code.",
            },
        },
        "conversion": {
            **schema_inputs,
            "schema_sha256": compute_schema_sha256(schema_inputs),
            "algorithm": [
                "Reduce the upstream 512-channel invariant atom embeddings to 128 channels by mean-pooling contiguous groups of four features.",
                "Map repo C/N/O/F one-hot atom columns to the reduced ISD-PaiNN element embeddings for atomic numbers 6/7/8/9.",
                "Map the repo 'other' atom column to the reduced upstream hydrogen embedding.",
                "Initialize repo P/S/Cl/Br/I atom columns with the mean reduced embedding over H/C/N/O/F because the upstream checkpoint does not contain those elements.",
                "Zero-initialize the repo handcrafted scalar atom-feature columns and coordinate/radial columns because the upstream checkpoint has no structurally compatible weights for those inputs.",
            ],
        },
        "notes": [
            "The normalized checkpoint is intentionally limited to the current PaiNNStageBackbone placeholder delivered in Story 4.7.",
            "This artifact is a deterministic conversion of a public pretrained source and is not derived from PFASBench training.",
            "Resume checkpoints supplied through top-level ckpt_path must still take precedence over this artifact.",
        ],
    }


def prepare_artifact(
    source_checkpoint_path: Path,
    source_config_path: Path,
    output_path: Path,
    metadata_output_path: Path | None,
    skip_download: bool,
) -> PreparedArtifact:
    """Prepare the repo-compatible checkpoint artifact and provenance metadata."""
    ensure_download(
        path=source_checkpoint_path,
        url=UPSTREAM_SOURCE_URL,
        expected_digests={
            "sha256": UPSTREAM_SOURCE_SHA256,
            "md5": UPSTREAM_SOURCE_MD5,
        },
        skip_download=skip_download,
    )
    ensure_download(
        path=source_config_path,
        url=UPSTREAM_CONFIG_URL,
        expected_digests={
            "sha256": UPSTREAM_CONFIG_SHA256,
            "md5": UPSTREAM_CONFIG_MD5,
        },
        skip_download=skip_download,
    )

    checkpoint = torch.load(source_checkpoint_path, map_location="cpu", weights_only=True)
    source_state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(source_state, dict):
        raise TypeError("Expected the upstream checkpoint to deserialize to a mapping.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    converted_state = build_repo_backbone_state_dict(source_state)
    torch.save(converted_state, output_path)

    metadata_path = metadata_output_path
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".metadata.yaml")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    prepared = PreparedArtifact(
        artifact_path=output_path,
        metadata_path=metadata_path,
        sha256=compute_digest(output_path, algorithm="sha256"),
        size_bytes=output_path.stat().st_size,
        source_sha256=compute_digest(source_checkpoint_path, algorithm="sha256"),
        config_sha256=compute_digest(source_config_path, algorithm="sha256"),
    )

    metadata = build_metadata(
        prepared=prepared,
        source_checkpoint_path=source_checkpoint_path,
        source_config_path=source_config_path,
        output_path=output_path,
    )
    metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")

    return prepared


def main() -> None:
    """Prepare the normalized repo-managed artifact and provenance metadata."""
    args = parse_args()
    prepared = prepare_artifact(
        source_checkpoint_path=args.source_checkpoint,
        source_config_path=args.source_config,
        output_path=args.output,
        metadata_output_path=args.metadata_output,
        skip_download=args.skip_download,
    )
    print(f"Wrote artifact: {prepared.artifact_path}")
    print(f"Wrote metadata: {prepared.metadata_path}")
    print(f"Artifact sha256: {prepared.sha256}")


if __name__ == "__main__":
    main()
