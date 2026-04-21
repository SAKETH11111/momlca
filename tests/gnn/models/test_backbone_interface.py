"""Tests for the shared backbone interface contract."""

from __future__ import annotations

from typing import get_type_hints

import pytest
import torch
from torch_geometric.data import Batch, Data

from gnn.models.backbones import BackboneOutput, BaseBackbone, PaiNNStageBackbone


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


def test_backbone_interface_annotations_are_explicit_for_static_analysis() -> None:
    """Forward/output_dim annotations should be explicit and importable."""
    base_forward_hints = get_type_hints(BaseBackbone.forward)
    stage_forward_hints = get_type_hints(PaiNNStageBackbone.forward)
    output_dim_hints = get_type_hints(BaseBackbone.output_dim.fget)  # type: ignore[arg-type]

    assert base_forward_hints["return"] is BackboneOutput
    assert stage_forward_hints["return"] is BackboneOutput
    assert output_dim_hints["return"] is int
