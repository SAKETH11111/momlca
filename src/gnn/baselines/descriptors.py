"""Molecular descriptor extraction for ML baselines.

Uses RDKit's Descriptors module to compute standard molecular descriptors
from SMILES strings for use with traditional ML models.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

from gnn.data.loaders import load_smiles
from gnn.exceptions import InvalidSMILESError

logger = logging.getLogger(__name__)


class MolecularDescriptorExtractor:
    """Extract RDKit molecular descriptors from SMILES strings.

    Computes 200+ standard descriptors including:
    - Physicochemical: MolWt, LogP, TPSA, MolMR
    - Topological: BertzCT, Chi indices, Kappa indices
    - Constitutional: NumHDonors, NumHAcceptors, NumRotatableBonds
    - Electronic: MaxPartialCharge, MinPartialCharge
    - And many more...

    Args:
        descriptor_names: Optional list of specific descriptors to compute.
            If None, computes all available descriptors.
        handle_errors: Strategy for invalid molecules ("nan", "raise", "skip")

    Example:
        >>> extractor = MolecularDescriptorExtractor()
        >>> smiles_list = ["CCO", "c1ccccc1", "CC(=O)O"]
        >>> X = extractor.fit_transform(smiles_list)
        >>> X.shape
        (3, 208)
    """

    def __init__(
        self,
        descriptor_names: list[str] | None = None,
        handle_errors: str = "nan",
    ) -> None:
        if handle_errors not in ("nan", "raise", "skip"):
            raise ValueError(f"handle_errors must be 'nan', 'raise', or 'skip', got {handle_errors}")

        self.handle_errors = handle_errors

        # Get all available descriptor names if not specified
        if descriptor_names is None:
            self._descriptor_names = [desc[0] for desc in Descriptors._descList]
        else:
            self._descriptor_names = list(descriptor_names)

        # Create descriptor calculator
        self._calculator = MoleculeDescriptors.MolecularDescriptorCalculator(
            self._descriptor_names
        )

        # For feature normalization (fitted during fit_transform)
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    @property
    def descriptor_names(self) -> list[str]:
        """List of descriptor names being computed."""
        return list(self._descriptor_names)

    @property
    def num_descriptors(self) -> int:
        """Number of descriptors being computed."""
        return len(self._descriptor_names)

    def transform(
        self,
        smiles_list: Sequence[str],
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract descriptors from SMILES strings.

        Args:
            smiles_list: List of SMILES strings
            normalize: Whether to apply z-score normalization (requires prior fit)

        Returns:
            Feature matrix of shape (n_samples, n_descriptors)
        """
        features = []
        skipped_indices = []

        for i, smiles in enumerate(smiles_list):
            try:
                mol = load_smiles(smiles)
                desc_values = self._calculator.CalcDescriptors(mol)
                features.append(list(desc_values))
            except (InvalidSMILESError, Exception) as e:
                if self.handle_errors == "raise":
                    raise
                elif self.handle_errors == "nan":
                    features.append([np.nan] * self.num_descriptors)
                    logger.warning("Failed to compute descriptors for SMILES %d: %s", i, e)
                else:  # skip
                    skipped_indices.append(i)
                    logger.warning("Skipping SMILES %d: %s", i, e)

        X = np.array(features, dtype=np.float64)

        # Replace inf values with nan
        X = np.where(np.isinf(X), np.nan, X)

        if normalize:
            if self._means is None or self._stds is None:
                raise RuntimeError(
                    "Cannot normalize: no normalization parameters. Call fit_transform() first."
                )
            X = (X - self._means) / (self._stds + 1e-8)

        return X

    def fit_transform(
        self,
        smiles_list: Sequence[str],
        normalize: bool = True,
    ) -> np.ndarray:
        """Fit normalizer and extract descriptors.

        Args:
            smiles_list: List of SMILES strings
            normalize: Whether to compute and apply normalization

        Returns:
            Feature matrix of shape (n_samples, n_descriptors)
        """
        X = self.transform(smiles_list, normalize=False)

        if normalize:
            # Compute stats ignoring NaN
            self._means = np.nanmean(X, axis=0)
            self._stds = np.nanstd(X, axis=0)
            X = (X - self._means) / (self._stds + 1e-8)

        return X

    def to_dataframe(
        self,
        smiles_list: Sequence[str],
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Extract descriptors as a pandas DataFrame.

        Useful for inspection and feature selection.

        Args:
            smiles_list: List of SMILES strings
            normalize: Whether to apply normalization

        Returns:
            DataFrame with descriptor values
        """
        X = self.transform(smiles_list, normalize=normalize)
        return pd.DataFrame(X, columns=self.descriptor_names)

    def get_valid_descriptor_mask(self, X: np.ndarray) -> np.ndarray:
        """Get mask of descriptors that have valid (non-NaN) values for all samples.

        Args:
            X: Feature matrix from transform()

        Returns:
            Boolean array of shape (n_descriptors,) indicating valid descriptors
        """
        return ~np.any(np.isnan(X), axis=0)
