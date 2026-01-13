"""Exception hierarchy for GNN framework."""


class GNNError(Exception):
    """Base exception for GNN framework."""


class FeaturizationError(GNNError):
    """Raised when molecular featurization fails."""


class InvalidSMILESError(FeaturizationError):
    """Raised when SMILES string cannot be parsed."""
