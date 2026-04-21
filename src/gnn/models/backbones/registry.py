"""Minimal backbone registry for config-driven model selection."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from gnn.models.backbones.base import BaseBackbone

_BACKBONE_REGISTRY: dict[str, type[BaseBackbone]] = {}
_BUILTIN_BACKBONE_MODULES: tuple[str, ...] = (
    "gnn.models.backbones.gin",
    "gnn.models.backbones.painn_stage",
)


def _normalize_backbone_name(name: str) -> str:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Backbone name must be a non-empty string.")
    return normalized_name


def _ensure_builtin_backbones_loaded() -> None:
    """Load built-in backbone modules so direct registry imports stay discoverable."""
    for module_name in _BUILTIN_BACKBONE_MODULES:
        importlib.import_module(module_name)


def register_backbone(name: str) -> Callable[[type[BaseBackbone]], type[BaseBackbone]]:
    """Register a backbone class under a stable string key."""
    normalized_name = _normalize_backbone_name(name)

    def decorator(backbone_cls: type[BaseBackbone]) -> type[BaseBackbone]:
        existing = _BACKBONE_REGISTRY.get(normalized_name)
        if existing is not None and existing is not backbone_cls:
            raise ValueError(
                f"Backbone '{normalized_name}' is already registered as {existing.__name__}."
            )
        _BACKBONE_REGISTRY[normalized_name] = backbone_cls
        return backbone_cls

    return decorator


def get_backbone_class(name: str) -> type[BaseBackbone]:
    """Return the registered backbone class for ``name``."""
    _ensure_builtin_backbones_loaded()
    normalized_name = _normalize_backbone_name(name)
    backbone_cls = _BACKBONE_REGISTRY.get(normalized_name)
    if backbone_cls is None:
        available = ", ".join(sorted(_BACKBONE_REGISTRY)) or "<empty>"
        raise KeyError(f"Backbone '{name}' is not registered. Available backbones: {available}.")
    return backbone_cls


def registered_backbones() -> dict[str, type[BaseBackbone]]:
    """Return a copy of current backbone registrations."""
    _ensure_builtin_backbones_loaded()
    return dict(_BACKBONE_REGISTRY)


__all__ = ["get_backbone_class", "register_backbone", "registered_backbones"]
