"""Tests for SDF and MOL file loading."""

from pathlib import Path

import pytest
from rdkit import Chem

from gnn.data.loaders.sdf import load_mol, load_sdf
from gnn.exceptions import InvalidFileError

# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent / "fixtures"


class TestLoadSDF:
    """Tests for load_sdf function."""

    def test_load_single_molecule_sdf(self):
        """Test loading SDF file with single molecule."""
        molecules = load_sdf(FIXTURES_DIR / "single_mol.sdf")

        assert isinstance(molecules, list)
        assert len(molecules) == 1
        assert isinstance(molecules[0], Chem.Mol)
        assert molecules[0].GetNumAtoms() > 0

    def test_load_multi_molecule_sdf(self):
        """Test loading SDF file with multiple molecules."""
        molecules = load_sdf(FIXTURES_DIR / "multi_mol.sdf")

        assert isinstance(molecules, list)
        assert len(molecules) == 3
        for mol in molecules:
            assert isinstance(mol, Chem.Mol)
            assert mol.GetNumAtoms() > 0

    def test_3d_coordinates_preserved(self):
        """Test that 3D coordinates are preserved when loading SDF."""
        molecules = load_sdf(FIXTURES_DIR / "single_mol.sdf")
        mol = molecules[0]

        # Check that molecule has conformer with 3D coordinates
        assert mol.GetNumConformers() > 0
        conf = mol.GetConformer()
        positions = conf.GetPositions()

        # Verify we have 3D coordinates (not all z-coords are zero)
        assert positions.shape[1] == 3  # x, y, z coordinates
        z_coords = positions[:, 2]
        assert any(abs(z) > 0.01 for z in z_coords), (
            "Expected 3D coordinates with non-zero z values"
        )

    def test_file_not_found_raises_error(self):
        """Test that missing file raises InvalidFileError."""
        with pytest.raises(InvalidFileError, match="File not found"):
            load_sdf(FIXTURES_DIR / "nonexistent.sdf")

    def test_invalid_file_handling(self):
        """Test that invalid SDF file raises InvalidFileError."""
        with pytest.raises(InvalidFileError, match="Failed to parse any molecules"):
            load_sdf(FIXTURES_DIR / "invalid.sdf")

    def test_empty_sdf_returns_empty_list(self):
        """Test that empty SDF file returns an empty list."""
        molecules = load_sdf(FIXTURES_DIR / "empty.sdf")
        assert molecules == []

    def test_accepts_string_path(self):
        """Test that function accepts string path."""
        molecules = load_sdf(str(FIXTURES_DIR / "single_mol.sdf"))
        assert len(molecules) == 1

    def test_accepts_path_object(self):
        """Test that function accepts Path object."""
        molecules = load_sdf(FIXTURES_DIR / "single_mol.sdf")
        assert len(molecules) == 1


class TestLoadMOL:
    """Tests for load_mol function."""

    def test_load_mol_file(self):
        """Test loading single molecule from MOL file."""
        mol = load_mol(FIXTURES_DIR / "single.mol")

        assert isinstance(mol, Chem.Mol)
        assert mol.GetNumAtoms() > 0

    def test_3d_coordinates_preserved(self):
        """Test that 3D coordinates are preserved when loading MOL."""
        mol = load_mol(FIXTURES_DIR / "single.mol")

        # Check that molecule has conformer with 3D coordinates
        assert mol.GetNumConformers() > 0
        conf = mol.GetConformer()
        positions = conf.GetPositions()

        # Verify we have 3D coordinates (not all z-coords are zero)
        assert positions.shape[1] == 3  # x, y, z coordinates
        z_coords = positions[:, 2]
        assert any(abs(z) > 0.01 for z in z_coords), (
            "Expected 3D coordinates with non-zero z values"
        )

    def test_file_not_found_raises_error(self):
        """Test that missing file raises InvalidFileError."""
        with pytest.raises(InvalidFileError, match="File not found"):
            load_mol(FIXTURES_DIR / "nonexistent.mol")

    def test_invalid_mol_file_raises_error(self):
        """Test that invalid MOL file raises InvalidFileError."""
        with pytest.raises(InvalidFileError, match="Failed to parse MOL file"):
            load_mol(FIXTURES_DIR / "invalid.sdf")  # Using invalid SDF as invalid MOL

    def test_accepts_string_path(self):
        """Test that function accepts string path."""
        mol = load_mol(str(FIXTURES_DIR / "single.mol"))
        assert isinstance(mol, Chem.Mol)

    def test_accepts_path_object(self):
        """Test that function accepts Path object."""
        mol = load_mol(FIXTURES_DIR / "single.mol")
        assert isinstance(mol, Chem.Mol)
