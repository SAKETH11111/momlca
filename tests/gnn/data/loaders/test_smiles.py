"""Tests for SMILES molecular loading."""

import pytest
from rdkit import Chem

from gnn import InvalidSMILESError
from gnn.data.loaders import load_smiles


class TestLoadSmiles:
    """Tests for the load_smiles function."""

    # Representative PFAS structures (10+ examples)
    PFAS_SMILES = [
        # Short-chain PFAS
        ("TFA", "C(=O)(C(C(F)(F)F)(F)F)O"),  # 2 carbons
        ("PFBA", "C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O"),  # 4 carbons
        ("PFPeA", "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O"),  # 5 carbons
        ("PFHxA", "C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O"),  # 6 carbons
        ("PFHpA", "C(=O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"),  # 7 carbons
        # Long-chain PFAS
        (
            "PFOA",
            "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        ),  # 8 carbons
        # Sulfonates
        (
            "PFOS",
            "C(C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)(F)F",
        ),
        ("PFBS", "C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)(F)F"),  # Short-chain sulfonate
        # Ether PFAS
        ("GenX", "C(=O)(C(C(C(OC(F)(F)F)(F)F)(F)F)(F)F)O"),  # Ether linkage
        # Additional test structures
        ("Perfluoromethane", "C(F)(F)(F)F"),  # Simplest fully fluorinated
        ("Perfluoroethane", "C(C(F)(F)F)(F)(F)F"),  # Two carbon chain
    ]

    @pytest.mark.parametrize("name,smiles", PFAS_SMILES)
    def test_valid_pfas_structures(self, name: str, smiles: str) -> None:
        """Test loading of valid PFAS structures."""
        mol = load_smiles(smiles)
        assert mol is not None
        assert isinstance(mol, Chem.Mol)
        # Verify molecule has atoms
        assert mol.GetNumAtoms() > 0

    def test_returns_rdkit_mol(self) -> None:
        """Test that function returns RDKit Mol object."""
        mol = load_smiles("CCO")  # Ethanol
        assert isinstance(mol, Chem.Mol)

    def test_canonicalization(self) -> None:
        """Test that molecules are canonicalized."""
        # Two different representations of ethanol
        smiles1 = "CCO"
        smiles2 = "OCC"

        mol1 = load_smiles(smiles1)
        mol2 = load_smiles(smiles2)

        # Canonical SMILES should be identical
        canonical1 = Chem.MolToSmiles(mol1, canonical=True)
        canonical2 = Chem.MolToSmiles(mol2, canonical=True)
        assert canonical1 == canonical2

    def test_salt_removal(self) -> None:
        """Test that salts are removed (largest fragment kept)."""
        # Sodium acetate - should keep acetic acid, remove Na+
        mol = load_smiles("[Na+].CC(=O)[O-]")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        # Should have acetic acid only (no Na)
        assert "Na" not in canonical
        assert mol.GetNumAtoms() == 4  # C, C, O, O

    def test_multi_fragment_keeps_largest(self) -> None:
        """Test that multi-fragment SMILES keeps the largest fragment."""
        mol = load_smiles("C.CCC")  # methane + propane -> keep propane
        canonical = Chem.MolToSmiles(mol, canonical=True)
        assert "." not in canonical
        assert canonical == "CCC"

    def test_invalid_smiles_raises_error(self) -> None:
        """Test that invalid SMILES raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError) as exc_info:
            load_smiles("not_a_valid_smiles")
        assert "not_a_valid_smiles" in str(exc_info.value)

    def test_empty_string_raises_error(self) -> None:
        """Test that empty string raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError) as exc_info:
            load_smiles("")
        assert "Empty SMILES" in str(exc_info.value)

    def test_whitespace_only_raises_error(self) -> None:
        """Test that whitespace-only string raises InvalidSMILESError."""
        with pytest.raises(InvalidSMILESError):
            load_smiles("   ")

    def test_malformed_smiles_raises_error(self) -> None:
        """Test various malformed SMILES strings."""
        invalid_smiles = [
            "C(",  # Unclosed parenthesis
            "C))",  # Extra parenthesis
            "XYZ",  # Invalid atoms
            "C===C",  # Invalid bond
        ]
        for smiles in invalid_smiles:
            with pytest.raises(InvalidSMILESError):
                load_smiles(smiles)

    def test_exception_contains_original_smiles(self) -> None:
        """Test that exception message contains the original SMILES."""
        bad_smiles = "invalid_smiles_123"
        with pytest.raises(InvalidSMILESError) as exc_info:
            load_smiles(bad_smiles)
        assert bad_smiles in str(exc_info.value)

    def test_exception_hierarchy(self) -> None:
        """Test that InvalidSMILESError has correct exception hierarchy."""
        from gnn import FeaturizationError, GNNError

        with pytest.raises(FeaturizationError):
            load_smiles("invalid")

        with pytest.raises(GNNError):
            load_smiles("invalid")
