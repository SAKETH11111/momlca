"""Tests for descriptor and fingerprint extraction."""

from __future__ import annotations

import numpy as np
import pytest
from rdkit import Chem

from gnn.baselines.descriptors import DescriptorExtractor, MolecularDescriptorExtractor
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
        assert len(extractor.descriptor_names) < len(DescriptorExtractor.available_descriptor_names())

    def test_molecule_first_extract(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        extractor = DescriptorExtractor(descriptor_names=["MolWt", "MolLogP", "TPSA"])
        features = extractor.extract(mol)

        assert features.shape == (3,)
        assert np.isfinite(features).all()

    def test_extract_batch_from_smiles(self) -> None:
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        extractor = DescriptorExtractor(descriptor_names=["MolWt", "MolLogP", "TPSA"])
        matrix = extractor.extract_from_smiles(smiles)

        assert matrix.shape == (3, 3)
        assert np.isfinite(matrix).all()

    def test_invalid_smiles_nan_strategy(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            handle_errors="nan",
            missing_value_strategy="nan",
        )
        matrix = extractor.extract_from_smiles(["CCO", "NOT_A_SMILES", "CCN"])

        assert matrix.shape == (3, 2)
        assert np.isnan(matrix[1]).all()
        assert np.isfinite(matrix[0]).all()
        assert np.isfinite(matrix[2]).all()

    def test_invalid_smiles_raise_strategy(self) -> None:
        extractor = DescriptorExtractor(handle_errors="raise")
        with pytest.raises(InvalidSMILESError):
            extractor.extract_from_smiles(["CCO", "NOT_A_SMILES"])

    def test_fingerprint_only_support(self) -> None:
        extractor = DescriptorExtractor(
            include_descriptors=False,
            fingerprint_type="maccs",
        )
        features = extractor.extract_from_smiles(["CCO"])

        assert features.shape == (1, 167)
        assert extractor.feature_names[0] == "maccs_bit_0"

    def test_combined_descriptor_and_fingerprint_features(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP"],
            fingerprint_type="morgan",
            fingerprint_size=32,
        )
        features = extractor.extract_from_smiles(["CCO"])

        assert features.shape == (1, 34)
        assert extractor.feature_names[:2] == ["MolWt", "MolLogP"]
        assert extractor.feature_names[-1] == "morgan_bit_31"

    def test_fit_transform_normalizes_and_imputes(self) -> None:
        extractor = DescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP", "TPSA"],
            missing_value_strategy="mean",
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
