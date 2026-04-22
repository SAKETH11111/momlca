"""Regression tests for shared registry utilities and adapters."""

from __future__ import annotations

from uuid import uuid4

import pytest
import torch
from torch import nn
from torch_geometric.data import Batch, Data, Dataset

from gnn.data.datasets import get_dataset_class, register_dataset
from gnn.models.backbones import get_backbone_class, register_backbone
from gnn.models.backbones.base import BackboneOutput, BaseBackbone
from gnn.models.heads import get_head_class, register_head
from gnn.utils.registry import ComponentRegistry, normalize_registry_name


def test_normalize_registry_name_trims_and_lowercases() -> None:
    """Registry keys should normalize deterministically across call sites."""
    assert normalize_registry_name("  PaiNN  ", component_type="Backbone") == "painn"


def test_normalize_registry_name_rejects_empty_values() -> None:
    """Blank registry keys should fail fast with an actionable error."""
    with pytest.raises(ValueError, match="non-empty string"):
        normalize_registry_name("   ", component_type="Dataset")


def test_component_registry_duplicate_key_rejects_different_class() -> None:
    """Shared registry utility should block duplicate normalized keys."""
    component_registry: ComponentRegistry[nn.Module] = ComponentRegistry(
        component_type="Head",
        plural_component_type="heads",
    )

    @component_registry.register(" custom_head ")
    class _CustomHeadA(nn.Module):
        pass

    with pytest.raises(ValueError, match="already registered"):

        @component_registry.register("CUSTOM_HEAD")
        class _CustomHeadB(nn.Module):
            pass


def test_component_registry_allows_reregistering_same_class() -> None:
    """Re-registering the exact same class under an equivalent key should be a no-op."""
    component_registry: ComponentRegistry[nn.Module] = ComponentRegistry(
        component_type="Head",
        plural_component_type="heads",
    )

    @component_registry.register(" property_like ")
    class _PropertyLikeHead(nn.Module):
        pass

    component_registry.register("PROPERTY_LIKE")(_PropertyLikeHead)

    assert component_registry.get("property_like") is _PropertyLikeHead


def test_backbone_adapter_duplicate_registration_uses_normalized_keys() -> None:
    """Backbone adapter should enforce duplicate checks after normalization."""
    runtime_name = f"runtime_backbone_{uuid4().hex}"

    @register_backbone(f" {runtime_name.upper()} ")
    class _RuntimeBackboneA(BaseBackbone):
        @property
        def output_dim(self) -> int:
            return 1

        def forward(self, batch: Batch) -> BackboneOutput:
            node_features = torch.ones((batch.x.shape[0], 1), dtype=torch.float32)
            graph_features = torch.ones((batch.num_graphs, 1), dtype=torch.float32)
            return {"node_features": node_features, "graph_features": graph_features}

    with pytest.raises(ValueError, match="already registered"):

        @register_backbone(runtime_name.lower())
        class _RuntimeBackboneB(BaseBackbone):
            @property
            def output_dim(self) -> int:
                return 1

            def forward(self, batch: Batch) -> BackboneOutput:
                node_features = torch.zeros((batch.x.shape[0], 1), dtype=torch.float32)
                graph_features = torch.zeros((batch.num_graphs, 1), dtype=torch.float32)
                return {"node_features": node_features, "graph_features": graph_features}

    assert get_backbone_class(runtime_name) is _RuntimeBackboneA


def test_head_adapter_duplicate_registration_uses_normalized_keys() -> None:
    """Head adapter should enforce duplicate checks after normalization."""
    runtime_name = f"runtime_head_{uuid4().hex}"

    @register_head(f" {runtime_name.upper()} ")
    class _RuntimeHeadA(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
            return self.linear(graph_features)

    with pytest.raises(ValueError, match="already registered"):

        @register_head(runtime_name.lower())
        class _RuntimeHeadB(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 1)

            def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
                return self.linear(graph_features)

    assert get_head_class(runtime_name) is _RuntimeHeadA


def test_dataset_adapter_duplicate_registration_uses_normalized_keys() -> None:
    """Dataset adapter should enforce duplicate checks after normalization."""
    runtime_name = f"runtime_dataset_{uuid4().hex}"

    @register_dataset(f" {runtime_name.upper()} ")
    class _RuntimeDatasetA(Dataset):
        def len(self) -> int:
            return 0

        def get(self, idx: int) -> Data:
            raise IndexError(idx)

    with pytest.raises(ValueError, match="already registered"):

        @register_dataset(runtime_name.lower())
        class _RuntimeDatasetB(Dataset):
            def len(self) -> int:
                return 0

            def get(self, idx: int) -> Data:
                raise IndexError(idx)

    assert get_dataset_class(runtime_name) is _RuntimeDatasetA
