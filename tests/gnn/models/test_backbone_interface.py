"""Tests for the shared backbone interface contract."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn.data.transforms.constants import ALLOWED_ATOMS, ATOM_FEATURE_DIM
from gnn.models.backbones import (
    BackboneOutput,
    BaseBackbone,
    GINBackbone,
    PaiNNBackbone,
    PaiNNStageBackbone,
    get_backbone_class,
    registered_backbones,
)


class _IncompleteBackbone(BaseBackbone):
    """Intentional abstract subclass used for ABC enforcement tests."""


def _make_batch() -> Batch:
    graph_one = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        pos=torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=torch.float32),
    )
    graph_two = Data(
        x=torch.tensor([[5.0, 6.0]], dtype=torch.float32),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 1), dtype=torch.float32),
        pos=torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32),
    )
    return Batch.from_data_list([graph_one, graph_two])


def _make_atom_features(atomic_numbers: list[int]) -> torch.Tensor:
    feature_tensor = torch.zeros((len(atomic_numbers), ATOM_FEATURE_DIM), dtype=torch.float32)
    atomic_number_to_index = {atomic_number: idx for idx, atomic_number in enumerate(ALLOWED_ATOMS)}
    for row_idx, atomic_number in enumerate(atomic_numbers):
        encoded_index = atomic_number_to_index.get(atomic_number, len(ALLOWED_ATOMS))
        feature_tensor[row_idx, encoded_index] = 1.0
    return feature_tensor


def _make_painn_batch() -> Batch:
    graph = Data(
        x=_make_atom_features([6, 8, 9]),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 1], [1, 0, 2, 1, 1, 2]],
            dtype=torch.long,
        ),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.1, 0.0, 0.0],
                [1.1, 0.7, 0.0],
            ],
            dtype=torch.float32,
        ),
    )
    return Batch.from_data_list([graph])


def _rotation_matrix_z(theta_radians: float) -> torch.Tensor:
    cos_theta = math.cos(theta_radians)
    sin_theta = math.sin(theta_radians)
    return torch.tensor(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )


def test_base_backbone_cannot_be_instantiated_without_required_contract() -> None:
    """Backbone implementers must provide output_dim and forward()."""
    with pytest.raises(TypeError):
        _IncompleteBackbone()


def test_painn_stage_backbone_returns_standardized_backbone_output() -> None:
    """The stage backbone should return node+graph features under the shared schema."""
    batch = _make_batch()
    backbone = PaiNNStageBackbone(hidden_channels=32, use_positions=True)

    outputs = backbone(batch)

    assert set(outputs) == {"node_features", "graph_features"}
    assert outputs["node_features"].shape == (batch.x.shape[0], 32)
    assert outputs["graph_features"].shape == (batch.num_graphs, 32)
    assert backbone.output_dim == 32


@pytest.mark.parametrize("pooling", ["sum", "mean"])
def test_gin_backbone_returns_standardized_backbone_output(pooling: str) -> None:
    """GIN backbone should return node+graph features under the shared schema."""
    batch = _make_batch()
    backbone = GINBackbone(input_dim=2, hidden_channels=32, num_layers=2, pooling=pooling)

    outputs = backbone(batch)

    assert set(outputs) == {"node_features", "graph_features"}
    assert outputs["node_features"].shape == (batch.x.shape[0], 32)
    assert outputs["graph_features"].shape == (batch.num_graphs, 32)
    assert backbone.output_dim == 32


def test_gin_backbone_output_dim_is_explicit_before_first_forward() -> None:
    """GIN output_dim should be stable prior to the first forward pass."""
    backbone = GINBackbone(input_dim=2, hidden_channels=48, num_layers=3)
    assert backbone.output_dim == 48


def test_gin_backbone_rejects_unknown_pooling_mode() -> None:
    """GIN should fail fast for unsupported global pooling selections."""
    with pytest.raises(ValueError, match="pooling"):
        GINBackbone(input_dim=2, pooling="max")


def test_builtin_backbones_are_registered_for_lookup_by_name() -> None:
    """Built-in backbones should be discoverable through the shared registry."""
    assert get_backbone_class("gin") is GINBackbone
    assert get_backbone_class("painn") is PaiNNBackbone
    assert get_backbone_class("painn_stage") is PaiNNStageBackbone
    assert registered_backbones()["gin"] is GINBackbone
    assert registered_backbones()["painn"] is PaiNNBackbone
    assert registered_backbones()["painn_stage"] is PaiNNStageBackbone


def test_registry_module_resolves_builtin_backbones_without_package_import_side_effects() -> None:
    """Direct registry imports should still discover built-in backbones."""
    repo_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json\n"
                "from gnn.models.backbones.registry import get_backbone_class, registered_backbones\n"
                "print(json.dumps({"
                '"gin": get_backbone_class("gin").__name__, '
                '"painn": get_backbone_class("painn").__name__, '
                '"painn_stage": get_backbone_class("painn_stage").__name__, '
                '"keys": sorted(registered_backbones())'
                "}))\n"
            ),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    payload = json.loads(result.stdout)
    assert payload == {
        "gin": "GINBackbone",
        "painn": "PaiNNBackbone",
        "painn_stage": "PaiNNStageBackbone",
        "keys": ["gin", "painn", "painn_stage"],
    }


def test_painn_backbone_returns_standardized_output_with_scalar_node_features() -> None:
    """PaiNN backbone should expose scalar node and graph features under the shared contract."""
    torch.manual_seed(7)
    batch = _make_painn_batch()
    backbone = PaiNNBackbone(hidden_channels=16, num_layers=2, num_rbf=8, cutoff=6.0)

    outputs = backbone(batch)

    assert set(outputs) == {"node_features", "graph_features"}
    assert outputs["node_features"].shape == (batch.x.shape[0], 16)
    assert outputs["graph_features"].shape == (batch.num_graphs, 16)
    assert backbone.output_dim == 16


def test_painn_backbone_requires_positions() -> None:
    """PaiNN should fail fast when 3D positions are missing."""
    batch = _make_painn_batch()
    delattr(batch, "pos")
    backbone = PaiNNBackbone(hidden_channels=16, num_layers=2, num_rbf=8, cutoff=6.0)

    with pytest.raises(ValueError, match="3D positions"):
        backbone(batch)


def test_painn_backbone_is_translation_invariant_for_outputs() -> None:
    """Translating all coordinates should not change PaiNN outputs."""
    torch.manual_seed(11)
    batch = _make_painn_batch()
    translated_batch = batch.clone()
    translated_batch.pos = batch.pos + torch.tensor([3.5, -2.0, 1.25], dtype=torch.float32)
    backbone = PaiNNBackbone(hidden_channels=16, num_layers=2, num_rbf=8, cutoff=6.0).eval()

    reference = backbone(batch)
    translated = backbone(translated_batch)

    assert torch.allclose(reference["node_features"], translated["node_features"], atol=1e-5)
    assert torch.allclose(reference["graph_features"], translated["graph_features"], atol=1e-5)


def test_painn_backbone_exported_features_are_rotation_invariant() -> None:
    """Exported PaiNN features should stay invariant under global rotations."""
    torch.manual_seed(13)
    batch = _make_painn_batch()
    rotation = _rotation_matrix_z(theta_radians=torch.pi / 2)
    rotated_batch = batch.clone()
    rotated_batch.pos = batch.pos @ rotation.T
    backbone = PaiNNBackbone(hidden_channels=16, num_layers=2, num_rbf=8, cutoff=6.0).eval()

    reference = backbone(batch)
    rotated = backbone(rotated_batch)

    assert torch.allclose(reference["node_features"], rotated["node_features"], atol=1e-4)
    assert torch.allclose(reference["graph_features"], rotated["graph_features"], atol=1e-4)


def test_painn_backbone_rejects_trainable_bessel_radial_basis() -> None:
    """Bessel radial bases should fail fast instead of silently ignoring trainability."""
    with pytest.raises(ValueError, match="rbf_trainable"):
        PaiNNBackbone(
            hidden_channels=16,
            num_layers=2,
            num_rbf=8,
            cutoff=6.0,
            radial_basis="bessel",
            rbf_trainable=True,
        )


def test_backbone_interface_annotations_are_explicit_for_static_analysis() -> None:
    """Forward/output_dim annotations should be explicit and importable."""
    base_forward_hints = get_type_hints(BaseBackbone.forward)
    stage_forward_hints = get_type_hints(PaiNNStageBackbone.forward)
    painn_forward_hints = get_type_hints(PaiNNBackbone.forward)
    gin_forward_hints = get_type_hints(GINBackbone.forward)
    output_dim_hints = get_type_hints(BaseBackbone.output_dim.fget)  # type: ignore[arg-type]

    assert base_forward_hints["return"] is BackboneOutput
    assert stage_forward_hints["return"] is BackboneOutput
    assert painn_forward_hints["return"] is BackboneOutput
    assert gin_forward_hints["return"] is BackboneOutput
    assert output_dim_hints["return"] is int
