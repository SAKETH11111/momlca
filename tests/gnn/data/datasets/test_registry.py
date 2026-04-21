"""Tests for dataset registry behavior."""

from __future__ import annotations

from uuid import uuid4

import pytest
from torch_geometric.data import Data, Dataset

from gnn.data.datasets import (
    PFASBenchDataset,
    get_dataset,
    get_dataset_class,
    register_dataset,
    registered_datasets,
)
from gnn.models.backbones import get_backbone_class
from gnn.models.heads import get_head
from gnn.utils import BACKBONE_REGISTRY, DATASET_REGISTRY, HEAD_REGISTRY


def test_builtin_pfasbench_dataset_is_registered() -> None:
    """Built-in dataset classes should be discoverable through the registry."""
    assert get_dataset("pfasbench") is PFASBenchDataset
    assert get_dataset_class("PFASBENCH") is PFASBenchDataset
    assert registered_datasets()["pfasbench"] is PFASBenchDataset


def test_dataset_registry_supports_runtime_extension_registration() -> None:
    """Custom datasets should be registerable without modifying package imports."""
    runtime_name = f"runtime_dataset_{uuid4().hex}"

    @register_dataset(runtime_name)
    class _RuntimeDataset(Dataset):
        def len(self) -> int:
            return 0

        def get(self, idx: int) -> Data:
            raise IndexError(idx)

    assert get_dataset(runtime_name.upper()) is _RuntimeDataset
    assert get_dataset_class(runtime_name) is _RuntimeDataset
    assert registered_datasets()[runtime_name] is _RuntimeDataset


def test_dataset_registry_unknown_name_errors_include_available_entries() -> None:
    """Unknown dataset lookups should provide available options in the error."""
    with pytest.raises(KeyError) as exc:
        get_dataset("not_a_real_dataset")

    message = str(exc.value)
    assert "Available datasets:" in message
    assert "pfasbench" in message


def test_shared_registry_dictionaries_expose_loaded_component_keys() -> None:
    """The shared registry module should expose dictionary-backed component stores."""
    get_backbone_class("gin")
    get_head("property")
    get_dataset("pfasbench")

    assert isinstance(BACKBONE_REGISTRY, dict)
    assert isinstance(HEAD_REGISTRY, dict)
    assert isinstance(DATASET_REGISTRY, dict)
    assert "gin" in BACKBONE_REGISTRY
    assert "property" in HEAD_REGISTRY
    assert "pfasbench" in DATASET_REGISTRY
