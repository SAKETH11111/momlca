"""GNN package for molecular property prediction."""

__version__ = "0.1.0"

from .exceptions import (
    FeaturizationError,
    GNNError,
    InvalidFileError,
    InvalidSMILESError,
)

__all__ = [
    "__version__",
    "GNNError",
    "FeaturizationError",
    "InvalidFileError",
    "InvalidSMILESError",
]
