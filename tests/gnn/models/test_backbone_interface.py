"""Tests for the shared backbone interface contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import get_type_hints

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn.models.backbones import (
    BackboneOutput,
    BaseBackbone,
    GINBackbone,
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
    assert get_backbone_class("painn") is PaiNNStageBackbone
    assert registered_backbones()["gin"] is GINBackbone
    assert registered_backbones()["painn"] is PaiNNStageBackbone


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
        "painn": "PaiNNStageBackbone",
        "keys": ["gin", "painn"],
    }


def test_backbone_interface_annotations_are_explicit_for_static_analysis() -> None:
    """Forward/output_dim annotations should be explicit and importable."""
    base_forward_hints = get_type_hints(BaseBackbone.forward)
    stage_forward_hints = get_type_hints(PaiNNStageBackbone.forward)
    gin_forward_hints = get_type_hints(GINBackbone.forward)
    output_dim_hints = get_type_hints(BaseBackbone.output_dim.fget)  # type: ignore[arg-type]

    assert base_forward_hints["return"] is BackboneOutput
    assert stage_forward_hints["return"] is BackboneOutput
    assert gin_forward_hints["return"] is BackboneOutput
    assert output_dim_hints["return"] is int
