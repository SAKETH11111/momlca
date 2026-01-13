"""Bond feature extraction for molecular graphs.

This module provides functions to extract bond-level features and edge indices
from RDKit Mol objects for use in Graph Neural Networks (PyTorch Geometric format).
"""

from typing import Any

import torch
from rdkit import Chem

from .constants import (
    BOND_FEATURE_DIM,
    BOND_STEREO_TYPES,
    BOND_TYPES,
    NUM_BOND_TYPES,
    NUM_STEREO_TYPES,
)


def get_edge_index(mol: Chem.Mol) -> torch.Tensor:
    """Get edge indices in PyTorch Geometric format (bidirectional).

    Creates an undirected graph by adding edges in both directions for each bond.

    Args:
        mol: RDKit Mol object

    Returns:
        Tensor of shape (2, num_edges * 2) with dtype long
        First row contains source node indices, second row contains target indices.

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CC")  # Ethane
        >>> edge_index = get_edge_index(mol)
        >>> edge_index.shape
        torch.Size([2, 2])  # One bond = 2 directed edges
    """
    edge_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Add both directions for undirected graph representation
        edge_list.append([i, j])
        edge_list.append([j, i])

    if not edge_list:
        return torch.zeros((2, 0), dtype=torch.long)

    return torch.tensor(edge_list, dtype=torch.long).T


def get_bond_features(mol: Chem.Mol) -> torch.Tensor:
    """Extract bond features from molecule.

    Features are duplicated for bidirectional edges to match edge_index ordering.

    Args:
        mol: RDKit Mol object

    Returns:
        Tensor of shape (num_edges * 2, num_bond_features) with dtype float32
        Each bond contributes two rows (one for each direction).

    Feature dimensions (total: 12):
        - Bond type one-hot: 5 (single, double, triple, aromatic, other)
        - Is conjugated: 1
        - Is in ring: 1
        - Stereo one-hot: 5 (none, E, Z, any, other)

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CC")  # Ethane
        >>> features = get_bond_features(mol)
        >>> features.shape
        torch.Size([2, 12])  # One bond = 2 directed edges
    """
    features = []
    for bond in mol.GetBonds():
        bond_feats = _get_single_bond_features(bond)
        # Duplicate features for bidirectional edges
        features.append(bond_feats)
        features.append(bond_feats)

    if not features:
        return torch.zeros((0, BOND_FEATURE_DIM), dtype=torch.float32)

    return torch.tensor(features, dtype=torch.float32)


def _get_single_bond_features(bond: Chem.Bond) -> list[float]:
    """Extract features for a single bond.

    Args:
        bond: RDKit Bond object

    Returns:
        List of feature values
    """
    bond_feats: list[float] = []

    # Bond type one-hot encoding
    bond_type = bond.GetBondType()
    bond_type_oh = _one_hot_encode(bond_type, BOND_TYPES, NUM_BOND_TYPES)
    bond_feats.extend(bond_type_oh)

    # Is conjugated
    bond_feats.append(1.0 if bond.GetIsConjugated() else 0.0)

    # Is in ring
    bond_feats.append(1.0 if bond.IsInRing() else 0.0)

    # Stereo one-hot encoding
    stereo = bond.GetStereo()
    stereo_oh = _one_hot_encode(stereo, BOND_STEREO_TYPES, NUM_STEREO_TYPES)
    bond_feats.extend(stereo_oh)

    return bond_feats


def _one_hot_encode(value: Any, allowed_values: list[Any], size: int) -> list[float]:
    """Create one-hot encoding for a value.

    Args:
        value: Value to encode
        allowed_values: List of allowed values (last position reserved for "other")
        size: Total size of one-hot vector (len(allowed_values) + 1 for "other")

    Returns:
        One-hot encoded list of floats
    """
    encoding = [0.0] * size
    if value in allowed_values:
        idx = allowed_values.index(value)
        encoding[idx] = 1.0
    else:
        # Unknown/other category (last position)
        encoding[-1] = 1.0
    return encoding
