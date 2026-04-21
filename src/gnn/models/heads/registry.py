"""Minimal head registry for config-driven model selection."""

from __future__ import annotations

import importlib
from collections.abc import Callable

from torch import nn

_HEAD_REGISTRY: dict[str, type[nn.Module]] = {}
_BUILTIN_HEAD_MODULES: tuple[str, ...] = ("gnn.models.heads.property",)


def _normalize_head_name(name: str) -> str:
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError("Head name must be a non-empty string.")
    return normalized_name


def _ensure_builtin_heads_loaded() -> None:
    """Load built-in head modules so direct registry imports stay discoverable."""
    for module_name in _BUILTIN_HEAD_MODULES:
        importlib.import_module(module_name)


def register_head(name: str) -> Callable[[type[nn.Module]], type[nn.Module]]:
    """Register a head class under a stable string key."""
    normalized_name = _normalize_head_name(name)

    def decorator(head_cls: type[nn.Module]) -> type[nn.Module]:
        existing = _HEAD_REGISTRY.get(normalized_name)
        if existing is not None and existing is not head_cls:
            raise ValueError(
                f"Head '{normalized_name}' is already registered as {existing.__name__}."
            )
        _HEAD_REGISTRY[normalized_name] = head_cls
        return head_cls

    return decorator


def get_head(name: str) -> type[nn.Module]:
    """Return the registered head class for ``name``."""
    _ensure_builtin_heads_loaded()
    normalized_name = _normalize_head_name(name)
    head_cls = _HEAD_REGISTRY.get(normalized_name)
    if head_cls is None:
        available = ", ".join(sorted(_HEAD_REGISTRY)) or "<empty>"
        raise KeyError(f"Head '{name}' is not registered. Available heads: {available}.")
    return head_cls


def get_head_class(name: str) -> type[nn.Module]:
    """Backward-compatible alias for ``get_head``."""
    return get_head(name)


def registered_heads() -> dict[str, type[nn.Module]]:
    """Return a copy of current head registrations."""
    _ensure_builtin_heads_loaded()
    return dict(_HEAD_REGISTRY)


__all__ = ["get_head", "get_head_class", "register_head", "registered_heads"]
