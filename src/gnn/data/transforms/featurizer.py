"""Combined molecular featurizer for GNN input.

This module provides a unified interface for extracting all molecular features
needed for Graph Neural Network input.
"""

from typing import TypedDict

import torch
from rdkit import Chem

from .atom_features import get_atom_features
from .bond_features import get_bond_features, get_edge_index
from .constants import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


class MoleculeFeatures(TypedDict):
    """Type definition for molecule feature dictionary.

    Attributes:
        x: Atom feature tensor of shape (num_atoms, 22)
        edge_index: Edge index tensor of shape (2, num_edges)
        edge_attr: Bond feature tensor of shape (num_edges, 12)
        num_nodes: Number of atoms in the molecule
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    num_nodes: int


class MoleculeFeaturizer:
    """Unified molecular featurizer for GNN input.

    This class combines atom and bond feature extraction into a single interface,
    producing tensors ready for PyTorch Geometric Data objects.

    Feature Dimensions:
        Atom features (x): 22 dimensions
            - Atomic number one-hot: 10 (C, N, O, F, P, S, Cl, Br, I, other)
            - Degree: 1
            - Formal charge: 1
            - Hybridization one-hot: 7 (S, SP, SP2, SP3, SP3D, SP3D2, other)
            - Is aromatic: 1
            - Is in ring: 1
            - Num Hs: 1

        Bond features (edge_attr): 12 dimensions
            - Bond type one-hot: 5 (single, double, triple, aromatic, other)
            - Is conjugated: 1
            - Is in ring: 1
            - Stereo one-hot: 5 (none, E, Z, any, other)

    Example:
        >>> from rdkit import Chem
        >>> featurizer = MoleculeFeaturizer()
        >>> mol = Chem.MolFromSmiles("C(F)(F)F")  # Trifluoromethane
        >>> features = featurizer.featurize(mol)
        >>> features["x"].shape
        torch.Size([4, 22])
        >>> features["edge_index"].shape
        torch.Size([2, 6])  # 3 bonds * 2 directions
        >>> features["edge_attr"].shape
        torch.Size([6, 12])
    """

    def __init__(self) -> None:
        """Initialize the featurizer."""
        self.atom_feature_dim = ATOM_FEATURE_DIM
        self.bond_feature_dim = BOND_FEATURE_DIM

    def featurize(self, mol: Chem.Mol) -> MoleculeFeatures:
        """Extract all features from a molecule.

        Args:
            mol: RDKit Mol object

        Returns:
            Dictionary containing:
                - x: Atom features tensor (num_atoms, 22)
                - edge_index: Edge indices tensor (2, num_edges)
                - edge_attr: Bond features tensor (num_edges, 12)
                - num_nodes: Number of atoms
        """
        x = get_atom_features(mol)
        edge_index = get_edge_index(mol)
        edge_attr = get_bond_features(mol)

        return MoleculeFeatures(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=mol.GetNumAtoms(),
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MoleculeFeaturizer("
            f"atom_dim={self.atom_feature_dim}, "
            f"bond_dim={self.bond_feature_dim})"
        )
