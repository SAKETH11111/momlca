"""Shared component registry utilities."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")

BACKBONE_REGISTRY: dict[str, type[object]] = {}
HEAD_REGISTRY: dict[str, type[object]] = {}
DATASET_REGISTRY: dict[str, type[object]] = {}


def normalize_registry_name(name: str, *, component_type: str) -> str:
    """Normalize a registry key and validate it is non-empty."""
    normalized_name = name.strip().lower()
    if not normalized_name:
        raise ValueError(f"{component_type} name must be a non-empty string.")
    return normalized_name


class ComponentRegistry(Generic[T]):
    """Registry for named component classes with optional lazy built-in imports."""

    def __init__(
        self,
        *,
        component_type: str,
        plural_component_type: str,
        builtin_modules: tuple[str, ...] = (),
        registry: dict[str, type[T]] | None = None,
    ) -> None:
        self._component_type = component_type
        self._plural_component_type = plural_component_type
        self._builtin_modules = builtin_modules
        self._registry: dict[str, type[T]] = registry if registry is not None else {}
        self._builtins_loaded = False

    def _normalize_name(self, name: str) -> str:
        return normalize_registry_name(name, component_type=self._component_type)

    def _ensure_builtin_modules_loaded(self) -> None:
        if self._builtins_loaded:
            return

        for module_name in self._builtin_modules:
            importlib.import_module(module_name)
        self._builtins_loaded = True

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        """Register a component class under ``name``."""
        normalized_name = self._normalize_name(name)

        def decorator(component_cls: type[T]) -> type[T]:
            existing = self._registry.get(normalized_name)
            if existing is not None and existing is not component_cls:
                raise ValueError(
                    f"{self._component_type} '{normalized_name}' is already registered as "
                    f"{existing.__name__}."
                )
            self._registry[normalized_name] = component_cls
            return component_cls

        return decorator

    def get(self, name: str) -> type[T]:
        """Return the registered class for ``name``."""
        self._ensure_builtin_modules_loaded()
        normalized_name = self._normalize_name(name)
        component_cls = self._registry.get(normalized_name)
        if component_cls is None:
            available = ", ".join(sorted(self._registry)) or "<empty>"
            raise KeyError(
                f"{self._component_type} '{name}' is not registered. Available "
                f"{self._plural_component_type}: {available}."
            )
        return component_cls

    def registered(self) -> dict[str, type[T]]:
        """Return a copy of current registrations."""
        self._ensure_builtin_modules_loaded()
        return dict(self._registry)

    @property
    def storage(self) -> dict[str, type[T]]:
        """Expose the underlying storage for dictionary-style consumers."""
        self._ensure_builtin_modules_loaded()
        return self._registry


__all__ = [
    "BACKBONE_REGISTRY",
    "ComponentRegistry",
    "DATASET_REGISTRY",
    "HEAD_REGISTRY",
    "normalize_registry_name",
]
