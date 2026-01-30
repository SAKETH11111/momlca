"""Tests for molecular descriptor extraction."""

import numpy as np
import pytest

from gnn.baselines.descriptors import MolecularDescriptorExtractor


class TestMolecularDescriptorExtractor:
    """Test suite for MolecularDescriptorExtractor."""

    def test_initialization_default(self) -> None:
        """Test extractor initializes with all descriptors."""
        extractor = MolecularDescriptorExtractor()
        # RDKit has 200+ descriptors
        assert extractor.num_descriptors > 200

    def test_initialization_custom_descriptors(self) -> None:
        """Test extractor with custom descriptor list."""
        desc_names = ["MolWt", "MolLogP", "TPSA"]
        extractor = MolecularDescriptorExtractor(descriptor_names=desc_names)
        assert extractor.num_descriptors == 3
        assert extractor.descriptor_names == desc_names

    def test_initialization_invalid_handle_errors(self) -> None:
        """Test that invalid handle_errors raises ValueError."""
        with pytest.raises(ValueError, match="handle_errors must be"):
            MolecularDescriptorExtractor(handle_errors="invalid")

    def test_transform_simple_molecules(self) -> None:
        """Test descriptor extraction for simple molecules."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        extractor = MolecularDescriptorExtractor()
        X = extractor.transform(smiles)

        assert X.shape[0] == 3
        assert X.shape[1] == extractor.num_descriptors
        assert not np.all(np.isnan(X))

    def test_transform_pfas_molecules(self) -> None:
        """Test descriptor extraction for PFAS molecules."""
        pfas_smiles = [
            "C(=O)(C(F)(F)F)O",  # TFA
            "C(=O)(C(C(F)(F)F)(F)F)O",  # PFPA
        ]
        extractor = MolecularDescriptorExtractor()
        X = extractor.transform(pfas_smiles)

        assert X.shape[0] == 2
        # Check that we got valid features
        assert not np.all(np.isnan(X))

    def test_transform_invalid_smiles_nan(self) -> None:
        """Test handling of invalid SMILES with nan strategy."""
        smiles = ["CCO", "INVALID_SMILES_XYZ", "c1ccccc1"]
        extractor = MolecularDescriptorExtractor(handle_errors="nan")
        X = extractor.transform(smiles)

        assert X.shape[0] == 3
        # Second row should be all NaN
        assert np.all(np.isnan(X[1]))
        # First and third rows should have valid values
        assert not np.all(np.isnan(X[0]))
        assert not np.all(np.isnan(X[2]))

    def test_transform_invalid_smiles_raise(self) -> None:
        """Test handling of invalid SMILES with raise strategy."""
        smiles = ["CCO", "INVALID_SMILES_XYZ"]
        extractor = MolecularDescriptorExtractor(handle_errors="raise")

        with pytest.raises(Exception):
            extractor.transform(smiles)

    def test_fit_transform_normalization(self) -> None:
        """Test that fit_transform normalizes features."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccc(O)cc1"]
        extractor = MolecularDescriptorExtractor()

        X = extractor.fit_transform(smiles, normalize=True)

        # After normalization, we should have stored means/stds
        assert extractor._means is not None
        assert extractor._stds is not None

        # Feature matrix should have expected shape
        assert X.shape == (5, extractor.num_descriptors)

    def test_transform_with_prior_fit(self) -> None:
        """Test that transform uses fitted normalization parameters."""
        train_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        test_smiles = ["CCCC", "c1ccc(O)cc1"]

        extractor = MolecularDescriptorExtractor()

        # Fit on train
        X_train = extractor.fit_transform(train_smiles, normalize=True)

        # Transform test with same normalization
        X_test = extractor.transform(test_smiles, normalize=True)

        assert X_train.shape[0] == 3
        assert X_test.shape[0] == 2
        assert X_train.shape[1] == X_test.shape[1]

    def test_transform_without_fit_raises(self) -> None:
        """Test that transform with normalize=True without fit raises error."""
        smiles = ["CCO", "c1ccccc1"]
        extractor = MolecularDescriptorExtractor()

        with pytest.raises(RuntimeError, match="no normalization parameters"):
            extractor.transform(smiles, normalize=True)

    def test_to_dataframe(self) -> None:
        """Test DataFrame output."""
        smiles = ["CCO", "c1ccccc1"]
        extractor = MolecularDescriptorExtractor()
        df = extractor.to_dataframe(smiles)

        assert len(df) == 2
        assert list(df.columns) == extractor.descriptor_names

    def test_get_valid_descriptor_mask(self) -> None:
        """Test getting mask of valid descriptors."""
        smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
        extractor = MolecularDescriptorExtractor()
        X = extractor.transform(smiles)

        mask = extractor.get_valid_descriptor_mask(X)

        assert mask.shape == (extractor.num_descriptors,)
        assert mask.dtype == bool
