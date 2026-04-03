"""Tests for descriptor and fingerprint extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from gnn.baselines.descriptors import (
    DescriptorExtractor,
    MolecularDescriptorExtractor,
    export_pfasbench_descriptors,
)
from gnn.exceptions import InvalidSMILESError


class TestDescriptorExtractor:
    """Test suite for descriptor extraction."""

    def test_initialization_default(self) -> None:
        extractor = DescriptorExtractor()
        assert extractor.num_descriptors > 200
        assert extractor.num_features == extractor.num_descriptors

    def test_alias_preserved(self) -> None:
        extractor = MolecularDescriptorExtractor(descriptor_names=["MolWt", "MolLogP"])
        assert isinstance(extractor, DescriptorExtractor)
        assert extractor.descriptor_names == ["MolWt", "MolLogP"]

    def test_subset_selection(self) -> None:
        extractor = DescriptorExtractor(descriptor_set="physicochemical")
        assert "MolWt" in extractor.descriptor_names
        assert len(extractor.descriptor_names) < len(
            DescriptorExtractor.available_descriptor_names()
        )

    def test_molecule_first_extract(self) -> None:
        mol = Chem.MolFromSmiles("C(=O)(C(F)(F)F)O")
        extractor = DescriptorExtractor(descriptor_names=["MolWt", "MolLogP", "TPSA"])
        features = extractor.extract(mol)

        assert list(features) == ["MolWt", "MolLogP", "TPSA"]
        assert all(np.isfinite(value) for value in features.values())

    def test_extract_batch_from_smiles(self) -> None:
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        extractor = DescriptorExtractor(descriptor_names=["MolWt", "MolLogP", "TPSA"])
        frame = extractor.extract_from_smiles(smiles)

        assert frame.shape == (3, 3)
        assert list(frame.columns) == ["MolWt", "MolLogP", "TPSA"]
        assert np.isfinite(frame.to_numpy()).all()

    def test_single_smiles_string_returns_single_row_frame(self) -> None:
        extractor = DescriptorExtractor(descriptor_names=["MolWt", "MolLogP"])
        frame = extractor.extract_from_smiles("CCO")

        assert frame.shape == (1, 2)
        assert list(frame.columns) == ["MolWt", "MolLogP"]

    def test_invalid_smiles_legacy_nan_strategy(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            handle_errors="nan",
            missing_value_strategy="nan",
        )
        frame = extractor.extract_from_smiles(["CCO", "NOT_A_SMILES", "CCN"])

        assert frame.shape == (3, 2)
        assert np.isnan(frame.iloc[1].to_numpy()).all()
        assert np.isfinite(frame.iloc[0].to_numpy()).all()
        assert np.isfinite(frame.iloc[2].to_numpy()).all()

    def test_invalid_smiles_raise_strategy(self) -> None:
        extractor = DescriptorExtractor(handle_errors="raise")
        with pytest.raises(InvalidSMILESError):
            extractor.extract_from_smiles(["CCO", "NOT_A_SMILES"])

    def test_fingerprint_only_support(self) -> None:
        extractor = DescriptorExtractor(
            include_descriptors=False,
            fingerprint_type="maccs",
        )
        frame = extractor.extract_from_smiles(["CCO"])

        assert frame.shape == (1, 167)
        assert extractor.feature_names[0] == "maccs_bit_0"

    def test_combined_descriptor_and_fingerprint_features(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            fingerprint_type="morgan",
            fingerprint_size=32,
        )
        frame = extractor.extract_from_smiles(["CCO"])

        assert frame.shape == (1, 34)
        assert extractor.feature_names[:2] == ["MolWt", "MolLogP"]
        assert extractor.feature_names[-1] == "morgan_bit_31"

    def test_handle_nan_modes_impute_mean_and_median(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            handle_nan="impute_mean",
        )
        mean_frame = extractor.extract_from_smiles(["CCO", "NOT_A_SMILES", "CCN"])
        median_frame = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            handle_nan="impute_median",
        ).extract_from_smiles(["CCO", "NOT_A_SMILES", "CCN"])

        assert np.isfinite(mean_frame.to_numpy()).all()
        assert np.isfinite(median_frame.to_numpy()).all()
        np.testing.assert_allclose(
            mean_frame.iloc[1].to_numpy(),
            mean_frame.iloc[[0, 2]].mean(axis=0).to_numpy(),
        )

    def test_handle_nan_zero_and_drop_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mol_a = Chem.MolFromSmiles("CCO")
        mol_b = Chem.MolFromSmiles("CCN")
        zero_extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            handle_nan="impute_zero",
        )
        zero_frame = zero_extractor.extract_from_smiles(["CCO", "NOT_A_SMILES"])
        assert np.allclose(zero_frame.iloc[1].to_numpy(), np.zeros(2))

        drop_extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            handle_nan="drop",
        )

        def fake_extract_single(molecule: Chem.Mol | None, *, failure_counts=None) -> np.ndarray:
            if molecule is mol_a:
                return np.array([1.0, np.nan], dtype=float)
            if molecule is mol_b:
                return np.array([2.0, 3.0], dtype=float)
            raise AssertionError("Unexpected molecule")

        monkeypatch.setattr(drop_extractor, "_extract_single", fake_extract_single)
        frame = drop_extractor.extract_batch([mol_a, mol_b])

        assert frame.shape == (2, 1)
        assert list(frame.columns) == ["MolWt"]

    def test_get_valid_descriptor_mask_ignores_fingerprint_columns_after_drop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mol_a = Chem.MolFromSmiles("CCO")
        mol_b = Chem.MolFromSmiles("CCN")
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            fingerprint_type="morgan",
            fingerprint_size=4,
            handle_nan="drop",
        )

        def fake_extract_single(molecule: Chem.Mol | None, *, failure_counts=None) -> np.ndarray:
            if molecule is mol_a:
                return np.array([1.0, np.nan, 0.0, 1.0, 0.0, 1.0], dtype=float)
            if molecule is mol_b:
                return np.array([2.0, 3.0, 1.0, 0.0, 1.0, 0.0], dtype=float)
            raise AssertionError("Unexpected molecule")

        monkeypatch.setattr(extractor, "_extract_single", fake_extract_single)
        frame = extractor.extract_batch([mol_a, mol_b])

        np.testing.assert_array_equal(
            extractor.get_valid_descriptor_mask(frame.to_numpy()),
            np.array([True, False]),
        )

    def test_fit_transform_normalizes_and_imputes(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP", "TPSA"],
            handle_nan="impute_mean",
        )
        train_features = extractor.fit_transform(["CCO", "CCN", "c1ccccc1"], normalize=True)
        test_features = extractor.transform(["CCCl", "CCBr"], normalize=True)

        assert train_features.shape == (3, 3)
        assert test_features.shape == (2, 3)
        assert np.isfinite(train_features).all()
        assert np.isfinite(test_features).all()

    def test_transform_without_fit_and_normalize_raises(self) -> None:
        extractor = DescriptorExtractor(descriptor_names=["MolWt"])
        with pytest.raises(RuntimeError, match="fit"):
            extractor.transform(["CCO"], normalize=True)

    def test_to_dataframe_uses_feature_names(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "TPSA"],
            fingerprint_type="morgan",
            fingerprint_size=8,
        )
        frame = extractor.to_dataframe(["CCO", "CCN"])

        assert list(frame.columns) == extractor.feature_names
        assert frame.shape == (2, 10)

    def test_get_feature_names_and_public_fingerprint_helper(self) -> None:
        mol = Chem.MolFromSmiles("C(=O)(C(C(F)(F)F)(F)F)O")
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            fingerprint_type="morgan",
            fingerprint_size=16,
        )
        fingerprint = extractor.compute_fingerprints(mol, fp_type="rdkit", nbits=16)

        assert fingerprint.shape == (16,)
        assert extractor.get_feature_names()[:2] == ["MolWt", "MolLogP"]

    def test_deterministic_output_on_pfas_molecules(self) -> None:
        smiles = [
            "C(=O)(C(F)(F)F)O",
            "C(=O)(C(C(F)(F)F)(F)F)O",
            "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O",
        ]
        extractor = DescriptorExtractor(descriptor_set="physicochemical", handle_nan="impute_mean")
        frame_one = extractor.extract_from_smiles(smiles)
        frame_two = extractor.extract_from_smiles(smiles)

        pd.testing.assert_frame_equal(frame_one, frame_two)

    def test_export_pfasbench_descriptors_writes_parquet(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "pfasbench.csv"
        input_path.write_text(
            "smiles,name,logS,logP,pKa\n"
            "C(=O)(C(F)(F)F)O,TFA,-0.5,0.5,0.5\n"
            "C(=O)(C(C(F)(F)F)(F)F)O,PFPA,-1.0,1.0,0.6\n"
        )

        output_path = export_pfasbench_descriptors(
            input_path,
            tmp_path / "descriptors.parquet",
            extractor=DescriptorExtractor(descriptor_names=["MolWt", "MolLogP"]),
        )

        assert output_path.exists()
        frame = pd.read_parquet(output_path)
        assert "MolWt" in frame.columns
        assert "MolLogP" in frame.columns
        assert len(frame) == 2

    def test_export_pfasbench_descriptors_preserves_invalid_smiles_as_nan(
        self,
        tmp_path: Path,
    ) -> None:
        input_path = tmp_path / "pfasbench.csv"
        input_path.write_text(
            "smiles,name\n"
            "C(=O)(C(F)(F)F)O,TFA\n"
            "NOT_A_SMILES,BAD\n"
        )

        output_path = export_pfasbench_descriptors(
            input_path,
            tmp_path / "descriptors.parquet",
            extractor=DescriptorExtractor(
                descriptor_names=["MolWt", "MolLogP"],
                handle_errors="nan",
                handle_nan="impute_mean",
            ),
        )

        frame = pd.read_parquet(output_path)
        invalid_row = frame.iloc[1]

        assert invalid_row["smiles"] == "NOT_A_SMILES"
        assert invalid_row[["MolWt", "MolLogP"]].isna().all()
