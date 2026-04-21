"""Dataset registry for config-driven dataset selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch_geometric.data import Dataset

from gnn.utils.registry import DATASET_REGISTRY, ComponentRegistry

_BUILTIN_DATASET_MODULES: tuple[str, ...] = ("gnn.data.datasets.pfasbench",)
_DATASET_REGISTRY = ComponentRegistry[Dataset](
    component_type="Dataset",
    plural_component_type="datasets",
    builtin_modules=_BUILTIN_DATASET_MODULES,
    registry=cast(dict[str, type[Dataset]], DATASET_REGISTRY),
)


def register_dataset(name: str) -> Callable[[type[Dataset]], type[Dataset]]:
    """Register a dataset class under a stable string key."""
    return _DATASET_REGISTRY.register(name)


def get_dataset(name: str) -> type[Dataset]:
    """Return the registered dataset class for ``name``."""
    return _DATASET_REGISTRY.get(name)


def get_dataset_class(name: str) -> type[Dataset]:
    """Backward-compatible alias for ``get_dataset``."""
    return get_dataset(name)


def registered_datasets() -> dict[str, type[Dataset]]:
    """Return a copy of current dataset registrations."""
    return _DATASET_REGISTRY.registered()


__all__ = [
    "get_dataset",
    "get_dataset_class",
    "register_dataset",
    "registered_datasets",
]
