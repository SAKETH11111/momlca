"""General utility helpers used across GNN modules."""

from gnn.utils.registry import (
    BACKBONE_REGISTRY,
    DATASET_REGISTRY,
    HEAD_REGISTRY,
    ComponentRegistry,
    normalize_registry_name,
)

__all__ = [
    "BACKBONE_REGISTRY",
    "ComponentRegistry",
    "DATASET_REGISTRY",
    "HEAD_REGISTRY",
    "normalize_registry_name",
]
