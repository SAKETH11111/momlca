"""Tests for bond feature extraction."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdchem

from gnn.data.transforms.bond_features import get_bond_features, get_edge_index
from gnn.data.transforms.constants import (
    BOND_FEATURE_DIM,
    BOND_IS_CONJUGATED_IDX,
    BOND_IS_IN_RING_IDX,
    BOND_STEREO_SLICE,
    BOND_STEREO_TYPES,
    BOND_TYPE_SLICE,
    BOND_TYPES,
)

# Test molecules with various bond types
TEST_SMILES = {
    "ethane": "CC",  # Single bond
    "ethene": "C=C",  # Double bond
    "ethyne": "C#C",  # Triple bond
    "benzene": "c1ccccc1",  # Aromatic bonds
    "TFA": "C(=O)(C(F)(F)F)O",  # Mixed bonds
    "cyclopropane": "C1CC1",  # Ring bonds
    "methane": "C",  # No bonds
    "conjugated": "C=CC=C",  # Conjugated system
}


class TestGetEdgeIndex:
    """Tests for get_edge_index function."""

    @pytest.mark.parametrize("name,smiles", list(TEST_SMILES.items()))
    def test_returns_correct_dtype(self, name: str, smiles: str):
        """Test that edge_index has correct dtype."""
        mol = Chem.MolFromSmiles(smiles)
        edge_index = get_edge_index(mol)

        assert edge_index.dtype == torch.long

    @pytest.mark.parametrize("name,smiles", list(TEST_SMILES.items()))
    def test_returns_correct_shape(self, name: str, smiles: str):
        """Test that edge_index has correct shape (2, num_edges)."""
        mol = Chem.MolFromSmiles(smiles)
        edge_index = get_edge_index(mol)

        num_bonds = mol.GetNumBonds()
        # Bidirectional: each bond creates 2 edges
        expected_edges = num_bonds * 2

        assert edge_index.shape == (2, expected_edges)

    def test_bidirectional_edges(self):
        """Test that edges are bidirectional."""
        mol = Chem.MolFromSmiles("CC")  # Ethane
        edge_index = get_edge_index(mol)

        # Should have [0, 1] and [1, 0]
        assert edge_index.shape == (2, 2)
        edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        assert (0, 1) in edges
        assert (1, 0) in edges

    def test_no_bonds_returns_empty(self):
        """Test that molecule with no bonds returns empty tensor."""
        mol = Chem.MolFromSmiles("C")  # Methane (single atom)
        edge_index = get_edge_index(mol)

        assert edge_index.shape == (2, 0)

    def test_edge_indices_are_valid(self):
        """Test that all edge indices are valid atom indices."""
        mol = Chem.MolFromSmiles("CCCC")  # Propane
        edge_index = get_edge_index(mol)

        num_atoms = mol.GetNumAtoms()
        assert edge_index.min() >= 0
        assert edge_index.max() < num_atoms


class TestGetBondFeatures:
    """Tests for get_bond_features function."""

    @pytest.mark.parametrize("name,smiles", list(TEST_SMILES.items()))
    def test_returns_correct_dtype(self, name: str, smiles: str):
        """Test that bond features have correct dtype."""
        mol = Chem.MolFromSmiles(smiles)
        features = get_bond_features(mol)

        assert features.dtype == torch.float32

    @pytest.mark.parametrize("name,smiles", list(TEST_SMILES.items()))
    def test_returns_correct_shape(self, name: str, smiles: str):
        """Test that bond features have correct shape."""
        mol = Chem.MolFromSmiles(smiles)
        features = get_bond_features(mol)

        num_bonds = mol.GetNumBonds()
        expected_edges = num_bonds * 2  # Bidirectional

        assert features.shape == (expected_edges, BOND_FEATURE_DIM)

    def test_matches_edge_index_length(self):
        """Test that bond features match edge_index in length."""
        mol = Chem.MolFromSmiles("CCCC")  # Butane
        edge_index = get_edge_index(mol)
        features = get_bond_features(mol)

        assert features.shape[0] == edge_index.shape[1]

    def test_single_bond_encoding(self):
        """Test that single bonds are correctly encoded."""
        mol = Chem.MolFromSmiles("CC")  # Ethane
        features = get_bond_features(mol)

        bond_type_onehot = features[0, BOND_TYPE_SLICE]
        assert bond_type_onehot[BOND_TYPES.index(rdchem.BondType.SINGLE)] == 1.0
        assert bond_type_onehot.sum() == 1.0

    def test_double_bond_encoding(self):
        """Test that double bonds are correctly encoded."""
        mol = Chem.MolFromSmiles("C=C")  # Ethene
        features = get_bond_features(mol)

        bond_type_onehot = features[0, BOND_TYPE_SLICE]
        assert bond_type_onehot[BOND_TYPES.index(rdchem.BondType.DOUBLE)] == 1.0
        assert bond_type_onehot.sum() == 1.0

    def test_triple_bond_encoding(self):
        """Test that triple bonds are correctly encoded."""
        mol = Chem.MolFromSmiles("C#C")  # Ethyne
        features = get_bond_features(mol)

        bond_type_onehot = features[0, BOND_TYPE_SLICE]
        assert bond_type_onehot[BOND_TYPES.index(rdchem.BondType.TRIPLE)] == 1.0
        assert bond_type_onehot.sum() == 1.0

    def test_aromatic_bond_encoding(self):
        """Test that aromatic bonds are correctly encoded."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        features = get_bond_features(mol)

        # All bonds in benzene should be aromatic
        aromatic_idx = BOND_TYPES.index(rdchem.BondType.AROMATIC)
        for i in range(features.shape[0]):
            bond_type_onehot = features[i, BOND_TYPE_SLICE]
            assert bond_type_onehot[aromatic_idx] == 1.0
            assert bond_type_onehot.sum() == 1.0

    def test_conjugation_feature(self):
        """Test that conjugation is correctly detected."""
        mol = Chem.MolFromSmiles("C=CC=C")  # 1,3-butadiene
        features = get_bond_features(mol)

        # At least some bonds should be conjugated
        conjugated_count = features[:, BOND_IS_CONJUGATED_IDX].sum().item()
        assert conjugated_count > 0

    def test_ring_membership_feature(self):
        """Test that ring membership is correctly detected for bonds."""
        mol = Chem.MolFromSmiles("C1CC1")  # Cyclopropane
        features = get_bond_features(mol)

        # All bonds in cyclopropane are in a ring
        for i in range(features.shape[0]):
            assert features[i, BOND_IS_IN_RING_IDX] == 1.0

    def test_no_bonds_returns_empty(self):
        """Test that molecule with no bonds returns empty tensor."""
        mol = Chem.MolFromSmiles("C")  # Methane
        features = get_bond_features(mol)

        assert features.shape == (0, BOND_FEATURE_DIM)

    def test_stereo_feature_default(self):
        """Test that stereo feature is encoded (default: none)."""
        mol = Chem.MolFromSmiles("CC")  # Ethane
        features = get_bond_features(mol)

        stereo_onehot = features[0, BOND_STEREO_SLICE]
        assert stereo_onehot[BOND_STEREO_TYPES.index(rdchem.BondStereo.STEREONONE)] == 1.0
        assert stereo_onehot.sum() == 1.0
