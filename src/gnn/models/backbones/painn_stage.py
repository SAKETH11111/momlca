"""Stage-specific PaiNN placeholder backbone."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from gnn.models.backbones.base import BackboneOutput, BaseBackbone
from gnn.models.backbones.registry import register_backbone


@register_backbone("painn_stage")
class PaiNNStageBackbone(BaseBackbone):
    """Legacy warm-start shim retained for the `model=painn_stage` preset.

    Fine-tune presets still rely on the historical stage checkpoint seam, while
    new 3D PaiNN runs should use `model=painn` and `PaiNNBackbone`.
    """

    def __init__(self, hidden_channels: int = 128, use_positions: bool = True) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_positions = use_positions
        self.node_projection = torch.nn.LazyLinear(hidden_channels)
        self.activation = torch.nn.SiLU()

    @property
    def output_dim(self) -> int:
        """Return graph-feature dimension used by downstream heads."""
        return self.hidden_channels

    def forward(self, batch: Batch) -> BackboneOutput:
        """Project atom features into a trainable graph embedding space."""
        node_features = batch.x
        if self.use_positions and getattr(batch, "pos", None) is not None:
            radial_distance = torch.linalg.vector_norm(batch.pos, dim=-1, keepdim=True)
            node_features = torch.cat((node_features, batch.pos, radial_distance), dim=-1)

        node_embeddings = self.activation(self.node_projection(node_features))
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(node_embeddings[mask].mean(dim=0))

        return {
            "node_features": node_embeddings,
            "graph_features": torch.stack(graph_features, dim=0),
        }
