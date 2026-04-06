"""Stage-specific PaiNN placeholder backbone."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch


class PaiNNStageBackbone(nn.Module):
    """Lightweight graph backbone that reserves a distinct `model=painn` path.

    This is intentionally a small shell so the Hydra config tree can
    expose a dedicated PaiNN branch before the full equivariant implementation is
    delivered.
    """

    def __init__(self, hidden_channels: int = 128, use_positions: bool = True) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.use_positions = use_positions
        self.node_projection = nn.LazyLinear(hidden_channels)
        self.activation = nn.SiLU()

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
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
