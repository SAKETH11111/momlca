"""Dataset classes for molecular property prediction.

This module provides PyTorch Geometric InMemoryDataset implementations
for various molecular benchmark datasets.
"""

from .pfasbench import PFASBenchDataset

__all__ = [
    "PFASBenchDataset",
]
