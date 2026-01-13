"""Tests for PFAS-family OOD splitting functionality."""

import logging

import numpy as np
import pytest
from rdkit import Chem

from gnn.data.splits.pfas_ood import (
    get_chain_length,
    get_headgroup,
    pfas_ood_split,
)


class TestGetChainLength:
    """Tests for get_chain_length function."""

    def test_tfa_returns_c2(self) -> None:
        """Test TFA (trifluoroacetic acid) returns C2."""
        mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        assert mol is not None
        chain_length = get_chain_length(mol)
        assert chain_length == "C2"

    def test_pfba_returns_c4(self) -> None:
        """Test PFBA (perfluorobutanoic acid) returns C4."""
        mol = Chem.MolFromSmiles("C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O")
        assert mol is not None
        chain_length = get_chain_length(mol)
        assert chain_length == "C4"

    def test_pfoa_returns_c8(self) -> None:
        """Test PFOA (perfluorooctanoic acid) returns C8."""
        mol = Chem.MolFromSmiles("C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O")
        assert mol is not None
        chain_length = get_chain_length(mol)
        assert chain_length == "C8"

    def test_pfbs_returns_c4(self) -> None:
        """Test PFBS (perfluorobutane sulfonate) returns C4."""
        mol = Chem.MolFromSmiles("C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)(F)F")
        assert mol is not None
        chain_length = get_chain_length(mol)
        assert chain_length == "C4"

    def test_unknown_for_no_carbons(self) -> None:
        """Test that molecules with no carbons return unknown."""
        mol = Chem.MolFromSmiles("[H][H]")  # H2
        assert mol is not None
        chain_length = get_chain_length(mol)
        assert chain_length == "unknown"

    def test_short_chain_c3(self) -> None:
        """Test a C3 PFAS chain."""
        mol = Chem.MolFromSmiles("C(=O)(C(C(F)(F)F)(F)F)O")  # PFPrA
        assert mol is not None
        chain_length = get_chain_length(mol)
        assert chain_length == "C3"


class TestGetHeadgroup:
    """Tests for get_headgroup function."""

    def test_carboxylate_pfoa(self) -> None:
        """Test PFOA returns carboxylate headgroup."""
        mol = Chem.MolFromSmiles("C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O")
        assert mol is not None
        headgroup = get_headgroup(mol)
        assert headgroup == "carboxylate"

    def test_carboxylate_tfa(self) -> None:
        """Test TFA returns carboxylate headgroup."""
        mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        assert mol is not None
        headgroup = get_headgroup(mol)
        assert headgroup == "carboxylate"

    def test_sulfonate_pfos(self) -> None:
        """Test PFOS returns sulfonate headgroup."""
        mol = Chem.MolFromSmiles(
            "C(C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)(F)F"
        )
        assert mol is not None
        headgroup = get_headgroup(mol)
        assert headgroup == "sulfonate"

    def test_sulfonate_pfbs(self) -> None:
        """Test PFBS returns sulfonate headgroup."""
        mol = Chem.MolFromSmiles("C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)(F)F")
        assert mol is not None
        headgroup = get_headgroup(mol)
        assert headgroup == "sulfonate"

    def test_alcohol_ftoh(self) -> None:
        """Test fluorotelomer alcohol returns alcohol headgroup."""
        mol = Chem.MolFromSmiles("C(C(C(C(F)(F)F)(F)F)(F)F)O")
        assert mol is not None
        headgroup = get_headgroup(mol)
        assert headgroup == "alcohol"

    def test_other_for_unknown_headgroup(self) -> None:
        """Test that unknown headgroups return 'other'."""
        mol = Chem.MolFromSmiles("C(F)(F)(F)C(F)(F)F")  # Hexafluoroethane (no headgroup)
        assert mol is not None
        headgroup = get_headgroup(mol)
        assert headgroup == "other"


class TestPfasOodSplit:
    """Tests for pfas_ood_split function."""

    @pytest.fixture
    def sample_pfas_smiles(self) -> list[str]:
        """Sample PFAS SMILES for testing splits."""
        return [
            # C2 carboxylates
            "C(=O)(C(F)(F)F)O",  # TFA - idx 0
            # C3 carboxylates
            "C(=O)(C(C(F)(F)F)(F)F)O",  # PFPrA - idx 1
            # C4 carboxylates
            "C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O",  # PFBA - idx 2
            # C4 sulfonates
            "C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)(F)F",  # PFBS - idx 3
            # C6 carboxylates
            "C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFHxA - idx 4
            # C8 carboxylates
            "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",  # PFOA - idx 5
            # C8 sulfonates
            "C(C(C(C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)(F)F",  # PFOS - idx 6
        ]

    def test_chain_length_ood_split_places_holdouts_in_test(
        self, sample_pfas_smiles: list[str]
    ) -> None:
        """Test that chain length holdout values are placed in test set."""
        train, val, test, stats = pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=["C2", "C3"],
            seed=42,
        )

        # C2 (idx 0) and C3 (idx 1) should be in test
        test_set = set(test.tolist())
        assert 0 in test_set, "C2 molecule (idx 0) should be in test set"
        assert 1 in test_set, "C3 molecule (idx 1) should be in test set"

        # They should NOT be in train or val
        train_val_set = set(train.tolist()) | set(val.tolist())
        assert 0 not in train_val_set, "C2 molecule should not be in train/val"
        assert 1 not in train_val_set, "C3 molecule should not be in train/val"

    def test_headgroup_ood_split_places_holdouts_in_test(
        self, sample_pfas_smiles: list[str]
    ) -> None:
        """Test that headgroup holdout values are placed in test set."""
        train, val, test, stats = pfas_ood_split(
            sample_pfas_smiles,
            holdout="headgroup",
            holdout_values=["sulfonate"],
            seed=42,
        )

        # Sulfonates (idx 3, 6) should be in test
        test_set = set(test.tolist())
        assert 3 in test_set, "PFBS (idx 3) should be in test set"
        assert 6 in test_set, "PFOS (idx 6) should be in test set"

        # They should NOT be in train or val
        train_val_set = set(train.tolist()) | set(val.tolist())
        assert 3 not in train_val_set, "PFBS should not be in train/val"
        assert 6 not in train_val_set, "PFOS should not be in train/val"

    def test_no_overlap_between_splits(self, sample_pfas_smiles: list[str]) -> None:
        """Test that no molecule appears in multiple splits."""
        train, val, test, _ = pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=["C2"],
            seed=42,
        )

        train_set = set(train.tolist())
        val_set = set(val.tolist())
        test_set = set(test.tolist())

        assert len(train_set & val_set) == 0, "Train and val should not overlap"
        assert len(train_set & test_set) == 0, "Train and test should not overlap"
        assert len(val_set & test_set) == 0, "Val and test should not overlap"

    def test_all_indices_covered(self, sample_pfas_smiles: list[str]) -> None:
        """Test that all molecule indices are in exactly one split."""
        train, val, test, _ = pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=["C2"],
            seed=42,
        )

        all_indices = set(train.tolist()) | set(val.tolist()) | set(test.tolist())
        expected = set(range(len(sample_pfas_smiles)))
        assert all_indices == expected

    def test_deterministic_with_same_seed(self, sample_pfas_smiles: list[str]) -> None:
        """Test that same seed produces identical splits."""
        train1, val1, test1, _ = pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=["C2"],
            seed=42,
        )
        train2, val2, test2, _ = pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=["C2"],
            seed=42,
        )

        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        np.testing.assert_array_equal(test1, test2)

    def test_returns_statistics_dict(self, sample_pfas_smiles: list[str]) -> None:
        """Test that split returns statistics dictionary."""
        _, _, _, stats = pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=["C2", "C3"],
            seed=42,
        )

        assert isinstance(stats, dict)
        assert "train" in stats
        assert "val" in stats
        assert "test" in stats
        # Test set should have C2 and C3 counts
        assert stats["test"].get("C2", 0) > 0 or stats["test"].get("C3", 0) > 0

    def test_logs_per_family_split_statistics(
        self, caplog: pytest.LogCaptureFixture, sample_pfas_smiles: list[str]
    ) -> None:
        caplog.set_level(logging.INFO, logger="gnn.data.splits.pfas_ood")

        pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=["C2", "C3"],
            seed=42,
        )

        # Ensure per-family counts are included in log output
        assert "Train:" in caplog.text
        assert "Val:" in caplog.text
        assert "Test (OOD):" in caplog.text
        assert "C2=" in caplog.text
        assert "C3=" in caplog.text

    def test_invalid_holdout_type_raises(self, sample_pfas_smiles: list[str]) -> None:
        """Test that invalid holdout type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown holdout type"):
            pfas_ood_split(
                sample_pfas_smiles,
                holdout="invalid_type",
                holdout_values=["C2"],
                seed=42,
            )

    def test_invalid_split_fractions_raise(self, sample_pfas_smiles: list[str]) -> None:
        with pytest.raises(ValueError, match="train_frac \\+ val_frac must equal 1.0"):
            pfas_ood_split(
                sample_pfas_smiles,
                holdout="chain_length",
                holdout_values=["C2"],
                seed=42,
                train_frac=0.9,
                val_frac=0.2,
            )

        with pytest.raises(ValueError, match="must be between 0 and 1"):
            pfas_ood_split(
                sample_pfas_smiles,
                holdout="chain_length",
                holdout_values=["C2"],
                seed=42,
                train_frac=-0.1,
                val_frac=1.1,
            )

    def test_empty_holdout_values_puts_nothing_in_test(self, sample_pfas_smiles: list[str]) -> None:
        """Test that empty holdout_values results in empty test set."""
        train, val, test, _ = pfas_ood_split(
            sample_pfas_smiles,
            holdout="chain_length",
            holdout_values=[],
            seed=42,
        )

        assert len(test) == 0, "Test set should be empty with no holdout values"
        assert len(train) + len(val) == len(sample_pfas_smiles)
