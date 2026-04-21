"""Dataset classes for molecular property prediction.

This module provides PyTorch Geometric InMemoryDataset implementations
for various molecular benchmark datasets.
"""

from .pfasbench import PFASBenchDataset
from .registry import (
    get_dataset,
    get_dataset_class,
    register_dataset,
    registered_datasets,
)

__all__ = [
    "PFASBenchDataset",
    "get_dataset",
    "get_dataset_class",
    "register_dataset",
    "registered_datasets",
]
