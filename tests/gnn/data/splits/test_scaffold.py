"""Tests for scaffold-based splitting functionality."""

import numpy as np
import pytest
from rdkit import Chem

import gnn.data.splits.scaffold as scaffold_module
from gnn.data.splits.scaffold import (
    get_scaffold,
    group_by_scaffold,
    scaffold_split,
)


class TestGetScaffold:
    """Tests for get_scaffold function."""

    def test_returns_scaffold_for_benzene(self) -> None:
        """Test that benzene returns a valid scaffold."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        assert mol is not None
        scaffold = get_scaffold(mol)
        # Benzene's scaffold is benzene itself
        assert scaffold == "c1ccccc1"

    def test_returns_scaffold_for_substituted_benzene(self) -> None:
        """Test that substituted benzene returns core scaffold."""
        # Toluene (methylbenzene) should return benzene scaffold
        mol = Chem.MolFromSmiles("Cc1ccccc1")
        assert mol is not None
        scaffold = get_scaffold(mol)
        assert scaffold == "c1ccccc1"

    def test_returns_scaffold_for_fused_ring(self) -> None:
        """Test scaffold extraction for naphthalene."""
        mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
        assert mol is not None
        scaffold = get_scaffold(mol)
        # Naphthalene's scaffold is naphthalene
        assert scaffold == "c1ccc2ccccc2c1"

    def test_returns_empty_for_acyclic_molecule(self) -> None:
        """Test that acyclic molecules return empty scaffold."""
        # Propane has no rings
        mol = Chem.MolFromSmiles("CCC")
        assert mol is not None
        scaffold = get_scaffold(mol)
        assert scaffold == ""

    def test_returns_empty_for_pfas_chain(self) -> None:
        """Test that PFAS linear chains return empty scaffold (no rings)."""
        # PFOA: perfluorooctanoic acid (linear chain)
        mol = Chem.MolFromSmiles("C(=O)(O)C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F")
        assert mol is not None
        scaffold = get_scaffold(mol)
        assert scaffold == ""

    def test_handles_pfas_with_ring(self) -> None:
        """Test PFAS molecules with aromatic rings return scaffolds."""
        # Fluorinated benzene derivative
        mol = Chem.MolFromSmiles("Fc1ccccc1")
        assert mol is not None
        scaffold = get_scaffold(mol)
        assert scaffold == "c1ccccc1"


class TestGroupByScaffold:
    """Tests for group_by_scaffold function."""

    def test_groups_same_scaffold_molecules(self) -> None:
        """Test molecules with same scaffold are grouped together."""
        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("Cc1ccccc1"),  # toluene
            Chem.MolFromSmiles("CCc1ccccc1"),  # ethylbenzene
        ]
        for m in mols:
            assert m is not None

        groups = group_by_scaffold(mols)
        # All should have benzene scaffold
        assert "c1ccccc1" in groups
        assert len(groups["c1ccccc1"]) == 3
        assert set(groups["c1ccccc1"]) == {0, 1, 2}

    def test_separates_different_scaffolds(self) -> None:
        """Test molecules with different scaffolds are separated."""
        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene
            Chem.MolFromSmiles("c1ccc2ccccc2c1"),  # naphthalene
        ]
        for m in mols:
            assert m is not None

        groups = group_by_scaffold(mols)
        assert len(groups) == 2
        assert groups["c1ccccc1"] == [0]
        assert groups["c1ccc2ccccc2c1"] == [1]

    def test_handles_acyclic_molecules_with_empty_key(self) -> None:
        """Test acyclic molecules are grouped under empty string key."""
        mols = [
            Chem.MolFromSmiles("CCC"),  # propane
            Chem.MolFromSmiles("CCCC"),  # butane
        ]
        for m in mols:
            assert m is not None

        groups = group_by_scaffold(mols)
        assert "" in groups
        assert len(groups[""]) == 2

    def test_mixed_molecules(self) -> None:
        """Test mixture of cyclic and acyclic molecules."""
        mols = [
            Chem.MolFromSmiles("c1ccccc1"),  # benzene - idx 0
            Chem.MolFromSmiles("CCC"),  # propane - idx 1
            Chem.MolFromSmiles("Cc1ccccc1"),  # toluene - idx 2
        ]
        for m in mols:
            assert m is not None

        groups = group_by_scaffold(mols)
        assert len(groups) == 2
        assert groups["c1ccccc1"] == [0, 2]
        assert groups[""] == [1]


class TestScaffoldSplit:
    """Tests for scaffold_split function."""

    @pytest.fixture
    def sample_smiles_list(self) -> list[str]:
        """Sample SMILES for testing splits."""
        return [
            # Benzene derivatives (same scaffold) - 4 molecules
            "c1ccccc1",  # benzene
            "Cc1ccccc1",  # toluene
            "CCc1ccccc1",  # ethylbenzene
            "OCc1ccccc1",  # benzyl alcohol
            # Naphthalene derivatives (same scaffold) - 3 molecules
            "c1ccc2ccccc2c1",  # naphthalene
            "Cc1ccc2ccccc2c1",  # methylnaphthalene
            "CCc1ccc2ccccc2c1",  # ethylnaphthalene
            # Acyclic (no scaffold) - 3 molecules
            "CCC",  # propane
            "CCCC",  # butane
            "CCCCC",  # pentane
        ]

    def test_split_fractions_must_sum_to_one(self, sample_smiles_list: list[str]) -> None:
        """Test that invalid fractions raise ValueError."""
        with pytest.raises(ValueError, match="must equal 1.0"):
            scaffold_split(
                sample_smiles_list,
                seed=42,
                train_frac=0.5,
                val_frac=0.3,
                test_frac=0.1,  # sums to 0.9
            )

    def test_returns_three_arrays(self, sample_smiles_list: list[str]) -> None:
        """Test that function returns train/val/test index arrays."""
        train, val, test = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        assert isinstance(train, np.ndarray)
        assert isinstance(val, np.ndarray)
        assert isinstance(test, np.ndarray)

    def test_no_overlap_between_splits(self, sample_smiles_list: list[str]) -> None:
        """Test that no molecule appears in multiple splits."""
        train, val, test = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        train_set = set(train.tolist())
        val_set = set(val.tolist())
        test_set = set(test.tolist())

        assert len(train_set & val_set) == 0, "Train and val should not overlap"
        assert len(train_set & test_set) == 0, "Train and test should not overlap"
        assert len(val_set & test_set) == 0, "Val and test should not overlap"

    def test_all_indices_covered(self, sample_smiles_list: list[str]) -> None:
        """Test that all molecule indices are in exactly one split."""
        train, val, test = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        all_indices = set(train.tolist()) | set(val.tolist()) | set(test.tolist())
        expected = set(range(len(sample_smiles_list)))
        assert all_indices == expected

    def test_deterministic_with_same_seed(self, sample_smiles_list: list[str]) -> None:
        """Test that same seed produces identical splits."""
        train1, val1, test1 = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        train2, val2, test2 = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        np.testing.assert_array_equal(train1, train2)
        np.testing.assert_array_equal(val1, val2)
        np.testing.assert_array_equal(test1, test2)

    def test_different_seed_produces_different_split(self, sample_smiles_list: list[str]) -> None:
        """Test that different seeds produce different splits."""
        train1, _, _ = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        train2, _, _ = scaffold_split(
            sample_smiles_list,
            seed=123,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        # Different seeds should (usually) produce different orderings
        # Note: with small datasets there's a chance they could be the same
        # but with 10 molecules it's unlikely
        assert not np.array_equal(train1, train2) or len(sample_smiles_list) < 5

    def test_scaffold_groups_stay_together(self, sample_smiles_list: list[str]) -> None:
        """Test that molecules with same scaffold are in the same split."""
        train, val, test = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        train_set = set(train.tolist())
        val_set = set(val.tolist())
        test_set = set(test.tolist())

        # Benzene derivatives (indices 0-3) should all be in same split
        benzene_indices = {0, 1, 2, 3}
        assert (
            benzene_indices <= train_set
            or benzene_indices <= val_set
            or benzene_indices <= test_set
        ), "Benzene derivatives should be in the same split"

        # Naphthalene derivatives (indices 4-6) should all be in same split
        naphthalene_indices = {4, 5, 6}
        assert (
            naphthalene_indices <= train_set
            or naphthalene_indices <= val_set
            or naphthalene_indices <= test_set
        ), "Naphthalene derivatives should be in the same split"

        # Acyclic molecules (indices 7-9) should all be in same split
        acyclic_indices = {7, 8, 9}
        assert (
            acyclic_indices <= train_set
            or acyclic_indices <= val_set
            or acyclic_indices <= test_set
        ), "Acyclic molecules should be in the same split"

    def test_respects_approximate_proportions(self, sample_smiles_list: list[str]) -> None:
        """Test that splits approximate target proportions."""
        train, val, test = scaffold_split(
            sample_smiles_list,
            seed=42,
            train_frac=0.8,
            val_frac=0.1,
            test_frac=0.1,
        )
        n = len(sample_smiles_list)
        # With scaffold constraints, proportions may not be exact
        # but train should be largest
        assert len(train) >= len(val)
        assert len(train) >= len(test)
        # All molecules should be assigned
        assert len(train) + len(val) + len(test) == n

    def test_accepts_list_of_smiles(self) -> None:
        """Test that function accepts a simple list of SMILES strings."""
        smiles = ["c1ccccc1", "Cc1ccccc1", "CCC"]
        train, val, test = scaffold_split(
            smiles,
            seed=42,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        assert len(train) + len(val) + len(test) == 3

    def test_accepts_train_val_test_aliases(self, sample_smiles_list: list[str]) -> None:
        """Test that the function accepts train/val/test keyword aliases."""
        train, val, test = scaffold_split(
            sample_smiles_list,
            seed=42,
            train=0.8,
            val=0.1,
            test=0.1,
        )
        assert len(train) + len(val) + len(test) == len(sample_smiles_list)

    def test_invalid_fraction_ranges_raise(self, sample_smiles_list: list[str]) -> None:
        """Test that per-fraction range validation is enforced."""
        with pytest.raises(ValueError, match="train_frac must be between 0 and 1"):
            scaffold_split(sample_smiles_list, train_frac=-0.1, val_frac=0.6, test_frac=0.5)

    def test_detects_overlap_via_validation(
        self, sample_smiles_list: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that runtime validation catches overlap between splits."""

        def fake_group_by_scaffold(mols: list[Chem.Mol]) -> dict[str, list[int]]:  # noqa: ARG001
            return {"a": [0, 1], "b": [1, 2], "c": [3, 4, 5, 6, 7, 8, 9]}

        monkeypatch.setattr(scaffold_module, "group_by_scaffold", fake_group_by_scaffold)
        with pytest.raises(ValueError, match="overlap detected"):
            scaffold_split(sample_smiles_list, seed=42, train_frac=0.8, val_frac=0.1, test_frac=0.1)

    def test_detects_missing_indices_via_validation(
        self, sample_smiles_list: list[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that runtime validation catches missing indices."""

        def fake_group_by_scaffold(mols: list[Chem.Mol]) -> dict[str, list[int]]:  # noqa: ARG001
            # omit index 9
            return {"a": [0, 1, 2], "b": [3, 4], "c": [5, 6, 7, 8]}

        monkeypatch.setattr(scaffold_module, "group_by_scaffold", fake_group_by_scaffold)
        with pytest.raises(ValueError, match="coverage invalid"):
            scaffold_split(sample_smiles_list, seed=42, train_frac=0.8, val_frac=0.1, test_frac=0.1)

    def test_empty_splits_are_int_dtype(self) -> None:
        """Test that empty splits still return integer dtype arrays."""
        # All share same scaffold -> entire group assigned to train, val/test empty.
        smiles = ["c1ccccc1", "Cc1ccccc1", "CCc1ccccc1"]
        train, val, test = scaffold_split(
            smiles, seed=42, train_frac=0.8, val_frac=0.1, test_frac=0.1
        )
        assert train.dtype == np.int64
        assert val.dtype == np.int64
        assert test.dtype == np.int64

    def test_handles_pfas_molecules(self) -> None:
        """Test scaffold split with PFAS-like molecules (limited scaffold diversity)."""
        pfas_smiles = [
            # Linear PFAS (no scaffold)
            "C(=O)(O)C(F)(F)F",  # TFA
            "C(=O)(O)C(C(F)(F)F)(F)F",  # PFPA
            "C(=O)(O)C(C(C(F)(F)F)(F)F)(F)F",  # PFHxA
            # Aromatic PFAS
            "Fc1ccccc1",  # fluorobenzene
            "Fc1ccc(F)cc1",  # difluorobenzene
        ]
        train, val, test = scaffold_split(
            pfas_smiles,
            seed=42,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        # All molecules should be assigned
        assert len(train) + len(val) + len(test) == 5
        # No overlap
        all_indices = set(train.tolist()) | set(val.tolist()) | set(test.tolist())
        assert all_indices == set(range(5))
