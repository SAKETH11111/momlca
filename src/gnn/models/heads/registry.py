"""Minimal head registry for config-driven model selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from torch import nn

from gnn.utils.registry import HEAD_REGISTRY, ComponentRegistry

_BUILTIN_HEAD_MODULES: tuple[str, ...] = (
    "gnn.models.heads.property",
    "gnn.models.heads.charge",
)
_HEAD_REGISTRY = ComponentRegistry[nn.Module](
    component_type="Head",
    plural_component_type="heads",
    builtin_modules=_BUILTIN_HEAD_MODULES,
    registry=cast(dict[str, type[nn.Module]], HEAD_REGISTRY),
)


def register_head(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Register a head class under a stable string key."""
    return _HEAD_REGISTRY.register(name)


def get_head(name: str) -> type[nn.Module]:
    """Return the registered head class for ``name``."""
    return _HEAD_REGISTRY.get(name)


def get_head_class(name: str) -> type[nn.Module]:
    """Backward-compatible alias for ``get_head``."""
    return get_head(name)


def registered_heads() -> dict[str, type[nn.Module]]:
    """Return a copy of current head registrations."""
    return _HEAD_REGISTRY.registered()


__all__ = ["get_head", "get_head_class", "register_head", "registered_heads"]
