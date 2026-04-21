"""Model building blocks for GNN training."""

from gnn.models.backbones import GINBackbone, PaiNNStageBackbone
from gnn.models.momlca_model import MoMLCAModel

__all__ = ["GINBackbone", "MoMLCAModel", "PaiNNStageBackbone"]
