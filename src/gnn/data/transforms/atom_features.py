"""Atom feature extraction for molecular graphs.

This module provides functions to extract atom-level features from RDKit Mol objects
for use in Graph Neural Networks.
"""

from typing import Any

import torch
from rdkit import Chem

from .constants import (
    ALLOWED_ATOMS,
    ATOM_FEATURE_DIM,
    HYBRIDIZATION_TYPES,
    NUM_ATOM_TYPES,
    NUM_HYBRIDIZATION_TYPES,
)


def get_atom_features(mol: Chem.Mol) -> torch.Tensor:
    """Extract atom features from molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        Tensor of shape (num_atoms, num_atom_features) with dtype float32

    Feature dimensions (total: 22):
        - Atomic number one-hot: 10 (C, N, O, F, P, S, Cl, Br, I, other)
        - Degree: 1
        - Formal charge: 1
        - Hybridization one-hot: 7 (S, SP, SP2, SP3, SP3D, SP3D2, other)
        - Is aromatic: 1
        - Is in ring: 1
        - Num Hs: 1

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("C(F)(F)F")  # Trifluoromethane
        >>> features = get_atom_features(mol)
        >>> features.shape
        torch.Size([4, 22])
    """
    features = []
    for atom in mol.GetAtoms():
        atom_feats = _get_single_atom_features(atom)
        features.append(atom_feats)

    if not features:
        # Handle empty molecule edge case
        return torch.zeros((0, ATOM_FEATURE_DIM), dtype=torch.float32)

    return torch.tensor(features, dtype=torch.float32)


def _get_single_atom_features(atom: Chem.Atom) -> list[float]:
    """Extract features for a single atom.

    Args:
        atom: RDKit Atom object

    Returns:
        List of feature values
    """
    atom_feats: list[float] = []

    # Atomic number one-hot encoding
    atom_num = atom.GetAtomicNum()
    atom_oh = _one_hot_encode(atom_num, ALLOWED_ATOMS, NUM_ATOM_TYPES)
    atom_feats.extend(atom_oh)

    # Degree (number of bonds)
    atom_feats.append(float(atom.GetDegree()))

    # Formal charge
    atom_feats.append(float(atom.GetFormalCharge()))

    # Hybridization one-hot encoding
    hyb = atom.GetHybridization()
    hyb_oh = _one_hot_encode(hyb, HYBRIDIZATION_TYPES, NUM_HYBRIDIZATION_TYPES)
    atom_feats.extend(hyb_oh)

    # Is aromatic
    atom_feats.append(1.0 if atom.GetIsAromatic() else 0.0)

    # Is in ring
    atom_feats.append(1.0 if atom.IsInRing() else 0.0)

    # Number of Hs (implicit + explicit)
    atom_feats.append(float(atom.GetTotalNumHs()))

    return atom_feats


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
