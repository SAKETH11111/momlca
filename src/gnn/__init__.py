"""GNN package for molecular property prediction."""

__version__ = "0.1.0"

from .exceptions import (
    FeaturizationError,
    GNNError,
    InvalidSMILESError,
)

__all__ = [
    "__version__",
    "GNNError",
    "FeaturizationError",
    "InvalidSMILESError",
]
