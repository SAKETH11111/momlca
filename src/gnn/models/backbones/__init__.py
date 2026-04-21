"""Backbone building blocks for GNN models."""

from gnn.models.backbones.base import BackboneOutput, BaseBackbone
from gnn.models.backbones.painn_stage import PaiNNStageBackbone

__all__ = ["BackboneOutput", "BaseBackbone", "PaiNNStageBackbone"]
