"""Minimal backbone registry for config-driven model selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from gnn.models.backbones.base import BaseBackbone
from gnn.utils.registry import BACKBONE_REGISTRY, ComponentRegistry

_BUILTIN_BACKBONE_MODULES: tuple[str, ...] = (
    "gnn.models.backbones.gin",
    "gnn.models.backbones.painn",
    "gnn.models.backbones.painn_stage",
)
_BACKBONE_REGISTRY = ComponentRegistry[BaseBackbone](
    component_type="Backbone",
    plural_component_type="backbones",
    builtin_modules=_BUILTIN_BACKBONE_MODULES,
    registry=cast(dict[str, type[BaseBackbone]], BACKBONE_REGISTRY),
)


def register_backbone(name: str) -> Callable[[type[BaseBackbone]], type[BaseBackbone]]:
    """Register a backbone class under a stable string key."""
    return _BACKBONE_REGISTRY.register(name)


def get_backbone(name: str) -> type[BaseBackbone]:
    """Return the registered backbone class for ``name``."""
    return _BACKBONE_REGISTRY.get(name)


def get_backbone_class(name: str) -> type[BaseBackbone]:
    """Backward-compatible alias for ``get_backbone``."""
    return get_backbone(name)


def registered_backbones() -> dict[str, type[BaseBackbone]]:
    """Return a copy of current backbone registrations."""
    return _BACKBONE_REGISTRY.registered()


__all__ = ["get_backbone", "get_backbone_class", "register_backbone", "registered_backbones"]
