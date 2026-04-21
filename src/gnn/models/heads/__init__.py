"""Prediction heads for graph-level tasks."""

from gnn.models.heads.property import PropertyHead
from gnn.models.heads.registry import get_head, get_head_class, register_head, registered_heads

__all__ = ["PropertyHead", "get_head", "get_head_class", "register_head", "registered_heads"]
