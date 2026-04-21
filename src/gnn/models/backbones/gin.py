"""2D GIN backbone implementation backed by PyTorch Geometric primitives."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

from gnn.models.backbones.base import BackboneOutput, BaseBackbone
from gnn.models.backbones.registry import register_backbone


def _resolve_pooling(pooling: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    normalized_pooling = pooling.strip().lower()
    if normalized_pooling == "sum":
        return global_add_pool
    if normalized_pooling == "mean":
        return global_mean_pool
    raise ValueError(f"Unsupported pooling mode '{pooling}'. Supported modes: sum, mean.")


@register_backbone("gin")
class GINBackbone(BaseBackbone):
    """Graph Isomorphism Network backbone for graph-level property prediction."""

    def __init__(
        self,
        input_dim: int = 22,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.0,
        train_eps: bool = False,
        pooling: str = "sum",
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.train_eps = train_eps
        self.pooling = pooling.strip().lower()
        self._pool = _resolve_pooling(self.pooling)
        self.input_projection = nn.Linear(input_dim, hidden_channels)
        self.convs = nn.ModuleList(
            [
                GINConv(
                    nn.Sequential(
                        nn.Linear(hidden_channels, hidden_channels),
                        nn.SiLU(),
                        nn.Linear(hidden_channels, hidden_channels),
                    ),
                    train_eps=train_eps,
                )
                for _ in range(num_layers)
            ]
        )

    @property
    def output_dim(self) -> int:
        """Return graph-feature dimension used by downstream heads."""
        return self.hidden_channels

    def forward(self, batch: Batch) -> BackboneOutput:
        """Encode a batch of molecular graphs into node and graph features."""
        node_features = F.silu(self.input_projection(batch.x))
        for conv in self.convs:
            node_features = conv(node_features, batch.edge_index)
            node_features = F.silu(node_features)
            node_features = F.dropout(node_features, p=self.dropout, training=self.training)

        graph_features = self._pool(node_features, batch.batch)
        return {
            "node_features": node_features,
            "graph_features": graph_features,
        }


__all__ = ["GINBackbone"]
