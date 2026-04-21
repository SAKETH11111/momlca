"""Test helpers for transfer-learning workflows."""

from __future__ import annotations

import torch
from torch_geometric.data import Batch

from gnn.models.backbones import BackboneOutput, BaseBackbone


class TinyBackbone(BaseBackbone):
    """Small trainable backbone used to exercise checkpoint loading paths."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.ReLU()

    @property
    def output_dim(self) -> int:
        """Return graph-feature dimension produced by the helper backbone."""
        return self.hidden_dim

    def forward(self, batch: Batch) -> BackboneOutput:
        """Project pooled node features into graph-level representations."""
        node_embeddings = self.activation(self.linear(batch.x))
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(node_embeddings[mask].mean(dim=0))
        return {
            "node_features": node_embeddings,
            "graph_features": torch.stack(graph_features, dim=0),
        }
