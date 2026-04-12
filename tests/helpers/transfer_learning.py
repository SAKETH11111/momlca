"""Test helpers for transfer-learning workflows."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Batch


class TinyBackbone(nn.Module):
    """Small trainable backbone used to exercise checkpoint loading paths."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 4) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Project pooled node features into graph-level representations."""
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            pooled_features = batch.x[mask].mean(dim=0)
            graph_features.append(self.activation(self.linear(pooled_features)))
        return {"graph_features": torch.stack(graph_features, dim=0)}

