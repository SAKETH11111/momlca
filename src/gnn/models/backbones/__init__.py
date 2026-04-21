"""Backbone building blocks for GNN models."""

from gnn.models.backbones.base import BackboneOutput, BaseBackbone
from gnn.models.backbones.gin import GINBackbone
from gnn.models.backbones.painn_stage import PaiNNStageBackbone
from gnn.models.backbones.registry import (
    get_backbone_class,
    register_backbone,
    registered_backbones,
)

__all__ = [
    "BackboneOutput",
    "BaseBackbone",
    "GINBackbone",
    "PaiNNStageBackbone",
    "get_backbone_class",
    "register_backbone",
    "registered_backbones",
]
