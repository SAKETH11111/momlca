"""Shared backbone interface contracts for graph encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypedDict

import torch
from torch import nn
from torch_geometric.data import Batch


class BackboneOutput(TypedDict):
    """Canonical backbone outputs used by model heads."""

    node_features: torch.Tensor
    graph_features: torch.Tensor


class BaseBackbone(nn.Module, ABC):
    """Abstract base class for all backbone implementations."""

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Return graph-feature dimension consumed by prediction heads."""

    @abstractmethod
    def forward(self, batch: Batch) -> BackboneOutput:
        """Encode a batch into node and graph feature representations."""


__all__ = ["BackboneOutput", "BaseBackbone"]
