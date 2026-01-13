"""Tests for 3D conformer generation utilities."""

import pytest
import torch
from rdkit import Chem

from gnn.data.transforms.conformer import generate_conformers, get_positions
from gnn.exceptions import FeaturizationError

# PFAS molecules for testing
PFAS_MOLECULES = {
    "TFA": "C(=O)(C(F)(F)F)O",  # Trifluoroacetic acid
    "PFOA": "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # Perfluorooctanoic acid
    "PFOS": "C(C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F",  # Perfluorooctanesulfonic acid
}


class TestGenerateConformers:
    """Tests for generate_conformers function."""

    def test_single_conformer_generation(self) -> None:
        """Test generating a single conformer."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol
        positions = generate_conformers(mol, num_conformers=1)

        assert len(positions) == 1
        assert isinstance(positions[0], torch.Tensor)

    def test_multiple_conformer_generation(self) -> None:
        """Test generating multiple conformers."""
        mol = Chem.MolFromSmiles("CCCC")  # Butane - has rotatable bonds
        positions = generate_conformers(mol, num_conformers=5)

        assert len(positions) == 5
        for pos in positions:
            assert isinstance(pos, torch.Tensor)

    def test_positions_tensor_shape(self) -> None:
        """Test that positions have correct shape (num_atoms, 3)."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol: 3 atoms in input Mol
        positions = generate_conformers(mol, num_conformers=1)

        assert positions[0].ndim == 2
        assert positions[0].shape[1] == 3  # x, y, z coordinates
        assert positions[0].shape[0] == mol.GetNumAtoms()

    def test_positions_tensor_dtype(self) -> None:
        """Test that positions are float32."""
        mol = Chem.MolFromSmiles("CCO")
        positions = generate_conformers(mol, num_conformers=1)

        assert positions[0].dtype == torch.float32

    def test_mmff_optimization_changes_geometry(self) -> None:
        """Test that MMFF optimization actually changes the geometry."""
        # Ethanol is simple but has bond lengths/angles that MMFF will adjust
        mol = Chem.MolFromSmiles("CCO")

        # Generate unoptimized conformers
        pos_unopt = generate_conformers(mol, num_conformers=1, optimize=False, random_seed=42)

        # Generate optimized conformers with same seed
        pos_opt = generate_conformers(mol, num_conformers=1, optimize=True, random_seed=42)

        # Positions should be different due to force field relaxation
        assert not torch.allclose(pos_unopt[0], pos_opt[0], atol=1e-4)

    def test_mmff_optimization_enabled(self) -> None:
        """Test conformer generation with MMFF optimization."""
        mol = Chem.MolFromSmiles("CCO")
        positions = generate_conformers(mol, num_conformers=1, optimize=True)

        assert len(positions) == 1
        assert positions[0].shape[1] == 3

    def test_mmff_optimization_disabled(self) -> None:
        """Test conformer generation without MMFF optimization."""
        mol = Chem.MolFromSmiles("CCO")
        positions = generate_conformers(mol, num_conformers=1, optimize=False)

        assert len(positions) == 1
        assert positions[0].shape[1] == 3

    def test_reproducibility_with_same_seed(self) -> None:
        """Test that same seed produces identical conformers."""
        mol = Chem.MolFromSmiles("CCCC")

        pos1 = generate_conformers(mol, num_conformers=1, random_seed=42)
        pos2 = generate_conformers(mol, num_conformers=1, random_seed=42)

        assert torch.allclose(pos1[0], pos2[0])

    def test_different_seeds_produce_different_conformers(self) -> None:
        """Test that different seeds produce different conformers."""
        mol = Chem.MolFromSmiles("CCCCCCCC")  # Octane - many conformations
        pos1 = generate_conformers(mol, num_conformers=1, random_seed=42)
        pos2 = generate_conformers(mol, num_conformers=1, random_seed=123)

        # Positions should be different (not exactly equal)
        assert not torch.allclose(pos1[0], pos2[0])

    def test_failure_raises_featurization_error_none_mol(self) -> None:
        """Test that None molecule raises FeaturizationError."""
        with pytest.raises(FeaturizationError, match="molecule is None"):
            generate_conformers(None)

    def test_failure_raises_featurization_error_single_atom(self) -> None:
        """Test that single-atom molecule raises FeaturizationError."""
        mol = Chem.MolFromSmiles("[He]")  # Single helium atom
        with pytest.raises(FeaturizationError, match="single-atom molecule"):
            generate_conformers(mol)

    @pytest.mark.parametrize("name,smiles", list(PFAS_MOLECULES.items()))
    def test_pfas_conformer_generation(self, name: str, smiles: str) -> None:
        """Test conformer generation for PFAS molecules."""
        mol = Chem.MolFromSmiles(smiles)
        positions = generate_conformers(mol, num_conformers=1)

        assert len(positions) == 1
        assert positions[0].ndim == 2
        assert positions[0].shape[1] == 3
        assert positions[0].shape[0] == mol.GetNumAtoms()

    def test_pfas_multiple_conformers(self) -> None:
        """Test generating multiple conformers for PFAS."""
        mol = Chem.MolFromSmiles(PFAS_MOLECULES["TFA"])
        positions = generate_conformers(mol, num_conformers=3)

        assert len(positions) == 3


class TestGetPositions:
    """Tests for get_positions convenience function."""

    def test_returns_single_tensor(self) -> None:
        """Test that get_positions returns a single tensor, not a list."""
        mol = Chem.MolFromSmiles("CCO")
        pos = get_positions(mol)

        assert isinstance(pos, torch.Tensor)
        assert pos.ndim == 2
        assert pos.shape[1] == 3

    def test_positions_shape(self) -> None:
        """Test correct shape for positions."""
        mol = Chem.MolFromSmiles("CCO")  # Ethanol: 3 atoms in input Mol
        pos = get_positions(mol)

        assert pos.shape == (3, 3)

    def test_positions_dtype(self) -> None:
        """Test positions are float32."""
        mol = Chem.MolFromSmiles("CCO")
        pos = get_positions(mol)

        assert pos.dtype == torch.float32

    def test_reproducibility(self) -> None:
        """Test reproducibility with same seed."""
        mol = Chem.MolFromSmiles("CCCC")

        pos1 = get_positions(mol, random_seed=42)
        pos2 = get_positions(mol, random_seed=42)

        assert torch.allclose(pos1, pos2)

    def test_failure_raises_featurization_error(self) -> None:
        """Test that invalid molecule raises FeaturizationError."""
        with pytest.raises(FeaturizationError):
            get_positions(None)

    @pytest.mark.parametrize("name,smiles", list(PFAS_MOLECULES.items()))
    def test_pfas_molecules(self, name: str, smiles: str) -> None:
        """Test get_positions works for PFAS molecules."""
        mol = Chem.MolFromSmiles(smiles)
        pos = get_positions(mol)

        assert isinstance(pos, torch.Tensor)
        assert pos.ndim == 2
        assert pos.shape[1] == 3
