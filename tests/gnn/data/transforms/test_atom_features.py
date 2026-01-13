"""Tests for atom feature extraction."""

import pytest
import torch
from rdkit import Chem

from gnn.data.transforms.atom_features import get_atom_features
from gnn.data.transforms.constants import (
    ALLOWED_ATOMS,
    ATOM_ATOMIC_NUMBER_SLICE,
    ATOM_DEGREE_IDX,
    ATOM_FEATURE_DIM,
    ATOM_FORMAL_CHARGE_IDX,
    ATOM_HYBRIDIZATION_SLICE,
    ATOM_IS_AROMATIC_IDX,
    ATOM_IS_IN_RING_IDX,
    ATOM_NUM_HS_IDX,
)

PFAS_SMILES = {
    "TFA": "C(=O)(C(F)(F)F)O",  # Trifluoroacetic acid
    "PFBA": "C(=O)(C(C(F)(F)F)(F)F)O",  # Perfluorobutanoic acid
    "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA
    "PFOS": "C(C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)(F)F",
    "benzene": "c1ccccc1",  # Aromatic ring
    "methane": "C",  # Single atom
    "charged": "[O-]C(=O)C(F)(F)F",  # Charged oxygen
}


class TestGetAtomFeatures:
    """Tests for get_atom_features function."""

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_returns_correct_shape(self, name: str, smiles: str):
        """Test that features have correct shape for various molecules."""
        mol = Chem.MolFromSmiles(smiles)
        features = get_atom_features(mol)

        assert features.shape == (mol.GetNumAtoms(), ATOM_FEATURE_DIM)

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_returns_float32_dtype(self, name: str, smiles: str):
        """Test that features have correct dtype."""
        mol = Chem.MolFromSmiles(smiles)
        features = get_atom_features(mol)

        assert features.dtype == torch.float32

    def test_carbon_atomic_number_encoding(self):
        """Test that carbon is correctly encoded in atomic number one-hot."""
        mol = Chem.MolFromSmiles("C")  # Methane
        features = get_atom_features(mol)

        # Carbon (atomic num 6) should be at index 0 in ALLOWED_ATOMS
        carbon_onehot = features[0, ATOM_ATOMIC_NUMBER_SLICE]
        assert carbon_onehot[ALLOWED_ATOMS.index(6)] == 1.0
        assert carbon_onehot.sum() == 1.0  # Only one position is 1

    def test_fluorine_atomic_number_encoding(self):
        """Test that fluorine is correctly encoded (common in PFAS)."""
        mol = Chem.MolFromSmiles("CF")  # Fluoromethane
        features = get_atom_features(mol)

        # Find the fluorine atom
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 9:  # Fluorine
                fluorine_onehot = features[i, ATOM_ATOMIC_NUMBER_SLICE]
                assert fluorine_onehot[ALLOWED_ATOMS.index(9)] == 1.0
                assert fluorine_onehot.sum() == 1.0

    def test_degree_feature(self):
        """Test that degree feature is correctly computed."""
        mol = Chem.MolFromSmiles("CC")  # Ethane
        features = get_atom_features(mol)

        # Both carbons have degree 1 (connected to each other)
        assert features[0, ATOM_DEGREE_IDX] == 1.0
        assert features[1, ATOM_DEGREE_IDX] == 1.0

    def test_formal_charge_feature(self):
        """Test that formal charge feature is correctly computed."""
        mol = Chem.MolFromSmiles("[O-]C(=O)C")  # Acetate ion
        features = get_atom_features(mol)

        # Find charged oxygen
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetFormalCharge() == -1:
                assert features[i, ATOM_FORMAL_CHARGE_IDX] == -1.0

    def test_aromatic_feature(self):
        """Test that aromaticity is correctly detected."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        features = get_atom_features(mol)

        # All carbons in benzene are aromatic
        for i in range(mol.GetNumAtoms()):
            assert features[i, ATOM_IS_AROMATIC_IDX] == 1.0

    def test_non_aromatic_feature(self):
        """Test that non-aromatic molecules have aromatic=0."""
        mol = Chem.MolFromSmiles("C(F)(F)F")  # Trifluoromethane
        features = get_atom_features(mol)

        # No atoms should be aromatic
        for i in range(mol.GetNumAtoms()):
            assert features[i, ATOM_IS_AROMATIC_IDX] == 0.0

    def test_ring_membership_feature(self):
        """Test that ring membership is correctly detected."""
        mol = Chem.MolFromSmiles("C1CC1")  # Cyclopropane
        features = get_atom_features(mol)

        # All carbons are in a ring
        for i in range(mol.GetNumAtoms()):
            assert features[i, ATOM_IS_IN_RING_IDX] == 1.0

    def test_num_hs_feature(self):
        """Test that hydrogen count is correctly computed."""
        mol = Chem.MolFromSmiles("C")  # Methane (C with 4 Hs)
        features = get_atom_features(mol)

        assert features[0, ATOM_NUM_HS_IDX] == 4.0  # Methane has 4 Hs

    def test_hybridization_encoding(self):
        """Test that hybridization is correctly encoded."""
        mol = Chem.MolFromSmiles("C=C")  # Ethene (sp2 carbons)
        features = get_atom_features(mol)

        # sp2 should be at index 2 within hybridization encoding
        for i in range(mol.GetNumAtoms()):
            hyb_onehot = features[i, ATOM_HYBRIDIZATION_SLICE]
            assert hyb_onehot[2] == 1.0  # SP2 is index 2
            assert hyb_onehot.sum() == 1.0

    def test_empty_molecule_handling(self):
        """Test that empty molecule returns empty tensor with correct shape."""
        # Create an empty molecule (no atoms)
        mol = Chem.RWMol()
        features = get_atom_features(mol)

        assert features.shape == (0, ATOM_FEATURE_DIM)
        assert features.dtype == torch.float32

    def test_sulfur_in_pfos(self):
        """Test that sulfur is correctly encoded in PFOS."""
        mol = Chem.MolFromSmiles("CS(=O)(=O)O")  # Methanesulfonic acid
        features = get_atom_features(mol)

        # Find sulfur atom
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 16:  # Sulfur
                sulfur_onehot = features[i, ATOM_ATOMIC_NUMBER_SLICE]
                assert sulfur_onehot[ALLOWED_ATOMS.index(16)] == 1.0
                assert sulfur_onehot.sum() == 1.0
