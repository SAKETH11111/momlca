"""Tests for combined MoleculeFeaturizer."""

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import rdchem

from gnn.data.transforms.constants import (
    ALLOWED_ATOMS,
    ATOM_FEATURE_DIM,
    ATOM_FORMAL_CHARGE_IDX,
    ATOM_IS_AROMATIC_IDX,
    ATOM_IS_IN_RING_IDX,
    BOND_FEATURE_DIM,
    BOND_IS_IN_RING_IDX,
    BOND_TYPES,
)
from gnn.data.transforms.featurizer import MoleculeFeaturizer

# PFAS test molecules
PFAS_SMILES = {
    "TFA": "C(=O)(C(F)(F)F)O",
    "PFBA": "C(=O)(C(C(F)(F)F)(F)F)O",
    "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
    "PFOS": "C(C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)(F)F",
    "PFBS": "C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)(F)F",
    "GenX": "C(=O)(C(C(C(OC(F)(F)F)(F)F)(F)F)(F)F)O",
}


class TestMoleculeFeaturizer:
    """Tests for MoleculeFeaturizer class."""

    @pytest.fixture
    def featurizer(self):
        """Create featurizer instance."""
        return MoleculeFeaturizer()

    def test_initialization(self, featurizer):
        """Test that featurizer initializes correctly."""
        assert featurizer.atom_feature_dim == ATOM_FEATURE_DIM
        assert featurizer.bond_feature_dim == BOND_FEATURE_DIM

    def test_repr(self, featurizer):
        """Test string representation."""
        repr_str = repr(featurizer)
        assert "MoleculeFeaturizer" in repr_str
        assert str(ATOM_FEATURE_DIM) in repr_str
        assert str(BOND_FEATURE_DIM) in repr_str

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_featurize_returns_dict(self, featurizer, name: str, smiles: str):
        """Test that featurize returns a dictionary with expected keys."""
        mol = Chem.MolFromSmiles(smiles)
        features = featurizer.featurize(mol)

        assert isinstance(features, dict)
        assert "x" in features
        assert "edge_index" in features
        assert "edge_attr" in features
        assert "num_nodes" in features

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_atom_features_shape(self, featurizer, name: str, smiles: str):
        """Test that atom features have correct shape."""
        mol = Chem.MolFromSmiles(smiles)
        features = featurizer.featurize(mol)

        assert features["x"].shape == (mol.GetNumAtoms(), ATOM_FEATURE_DIM)

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_edge_index_shape(self, featurizer, name: str, smiles: str):
        """Test that edge_index has correct shape."""
        mol = Chem.MolFromSmiles(smiles)
        features = featurizer.featurize(mol)

        num_bonds = mol.GetNumBonds()
        assert features["edge_index"].shape == (2, num_bonds * 2)

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_edge_attr_shape(self, featurizer, name: str, smiles: str):
        """Test that edge_attr has correct shape."""
        mol = Chem.MolFromSmiles(smiles)
        features = featurizer.featurize(mol)

        num_bonds = mol.GetNumBonds()
        assert features["edge_attr"].shape == (num_bonds * 2, BOND_FEATURE_DIM)

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_num_nodes_correct(self, featurizer, name: str, smiles: str):
        """Test that num_nodes matches molecule atom count."""
        mol = Chem.MolFromSmiles(smiles)
        features = featurizer.featurize(mol)

        assert features["num_nodes"] == mol.GetNumAtoms()

    @pytest.mark.parametrize("name,smiles", list(PFAS_SMILES.items()))
    def test_tensor_dtypes(self, featurizer, name: str, smiles: str):
        """Test that tensors have correct dtypes."""
        mol = Chem.MolFromSmiles(smiles)
        features = featurizer.featurize(mol)

        assert features["x"].dtype == torch.float32
        assert features["edge_index"].dtype == torch.long
        assert features["edge_attr"].dtype == torch.float32

    def test_edge_attr_matches_edge_index(self, featurizer):
        """Test that edge_attr length matches edge_index."""
        mol = Chem.MolFromSmiles("CCCC")  # Butane
        features = featurizer.featurize(mol)

        assert features["edge_attr"].shape[0] == features["edge_index"].shape[1]

    def test_single_atom_molecule(self, featurizer):
        """Test featurization of single atom molecule (no bonds)."""
        mol = Chem.MolFromSmiles("C")  # Methane
        features = featurizer.featurize(mol)

        assert features["x"].shape == (1, ATOM_FEATURE_DIM)
        assert features["edge_index"].shape == (2, 0)
        assert features["edge_attr"].shape == (0, BOND_FEATURE_DIM)
        assert features["num_nodes"] == 1

    def test_aromatic_molecule(self, featurizer):
        """Test featurization of aromatic molecule."""
        mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
        features = featurizer.featurize(mol)

        # Check that aromatic feature is set
        assert features["x"][:, ATOM_IS_AROMATIC_IDX].sum() == 6  # All 6 carbons aromatic

        # Check that bonds are aromatic
        aromatic_bond_idx = BOND_TYPES.index(rdchem.BondType.AROMATIC)
        assert features["edge_attr"][:, aromatic_bond_idx].sum() == 12  # 6 bonds * 2

    def test_charged_molecule(self, featurizer):
        """Test featurization of molecule with formal charge."""
        mol = Chem.MolFromSmiles("[O-]C(=O)C(F)(F)F")  # Trifluoroacetate
        features = featurizer.featurize(mol)

        # Should have one atom with charge -1
        assert features["x"][:, ATOM_FORMAL_CHARGE_IDX].min().item() == -1.0

    def test_pfas_has_many_fluorines(self, featurizer):
        """Test that PFAS molecules correctly identify fluorine atoms."""
        mol = Chem.MolFromSmiles("C(F)(F)F")  # Trifluoromethane
        features = featurizer.featurize(mol)

        fluorine_count = features["x"][:, ALLOWED_ATOMS.index(9)].sum().item()
        assert fluorine_count == 3  # 3 fluorines

    def test_multiple_bond_types(self, featurizer):
        """Test molecule with multiple bond types."""
        mol = Chem.MolFromSmiles("C(=O)(C)O")  # Acetic acid (single and double bonds)
        features = featurizer.featurize(mol)

        # Check we have both single and double bonds
        single_bond_idx = BOND_TYPES.index(rdchem.BondType.SINGLE)
        double_bond_idx = BOND_TYPES.index(rdchem.BondType.DOUBLE)

        has_single = features["edge_attr"][:, single_bond_idx].sum() > 0
        has_double = features["edge_attr"][:, double_bond_idx].sum() > 0

        assert has_single
        assert has_double

    def test_ring_molecule(self, featurizer):
        """Test molecule with ring structure."""
        mol = Chem.MolFromSmiles("C1CC1")  # Cyclopropane
        features = featurizer.featurize(mol)

        # Check atoms are in ring
        assert features["x"][:, ATOM_IS_IN_RING_IDX].sum() == 3  # All 3 atoms in ring

        # Check bonds are in ring
        assert features["edge_attr"][:, BOND_IS_IN_RING_IDX].sum() == 6  # 3 bonds * 2 directions
