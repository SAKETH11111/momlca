"""Descriptor and fingerprint extraction for tabular baseline models."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

from gnn.data.loaders.smiles import load_smiles
from gnn.exceptions import InvalidSMILESError

logger = logging.getLogger(__name__)

DescriptorSet = Literal["all", "2d", "physicochemical", "topological"]
FingerprintType = Literal["morgan", "rdkit", "maccs"] | None
HandleErrors = Literal["nan", "raise", "skip"]
MissingValueStrategy = Literal["nan", "mean", "zero"]

PHYSICOCHEMICAL_DESCRIPTOR_NAMES = [
    "MolWt",
    "ExactMolWt",
    "HeavyAtomMolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "FractionCSP3",
    "HeavyAtomCount",
    "NHOHCount",
    "NOCount",
    "NumAliphaticCarbocycles",
    "NumAliphaticHeterocycles",
    "NumAliphaticRings",
    "NumAromaticCarbocycles",
    "NumAromaticHeterocycles",
    "NumAromaticRings",
    "NumHAcceptors",
    "NumHDonors",
    "NumHeteroatoms",
    "NumRotatableBonds",
    "RingCount",
]

TOPOLOGICAL_DESCRIPTOR_NAMES = [
    "BertzCT",
    "BalabanJ",
    "Chi0",
    "Chi0n",
    "Chi0v",
    "Chi1",
    "Chi1n",
    "Chi1v",
    "Chi2n",
    "Chi2v",
    "Chi3n",
    "Chi3v",
    "Chi4n",
    "Chi4v",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
]

_DESCRIPTOR_FUNCTIONS = dict(Descriptors.descList)
_ALL_DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
_SUBSET_MAP: dict[DescriptorSet, list[str]] = {
    "all": _ALL_DESCRIPTOR_NAMES,
    "2d": _ALL_DESCRIPTOR_NAMES,
    "physicochemical": PHYSICOCHEMICAL_DESCRIPTOR_NAMES,
    "topological": TOPOLOGICAL_DESCRIPTOR_NAMES,
}


class DescriptorExtractor:
    """Extract RDKit descriptors and optional fingerprints.

    The public API is molecule-first:
    - :meth:`extract` for one ``Chem.Mol``
    - :meth:`extract_batch` for multiple molecules
    - :meth:`extract_from_smiles` for SMILES convenience

    The class also keeps the older ``transform`` / ``fit_transform`` interface so
    existing training code can continue to work while using a more explicit API.
    """

    def __init__(
        self,
        descriptor_names: Sequence[str] | None = None,
        descriptor_set: DescriptorSet = "all",
        fingerprint_type: FingerprintType = None,
        fingerprint_size: int = 2048,
        fingerprint_radius: int = 2,
        include_descriptors: bool = True,
        handle_errors: HandleErrors = "nan",
        missing_value_strategy: MissingValueStrategy = "mean",
    ) -> None:
        if handle_errors not in {"nan", "raise", "skip"}:
            raise ValueError("handle_errors must be one of: 'nan', 'raise', 'skip'")
        if missing_value_strategy not in {"nan", "mean", "zero"}:
            raise ValueError("missing_value_strategy must be one of: 'nan', 'mean', 'zero'")
        if not include_descriptors and fingerprint_type is None:
            raise ValueError("At least one feature source must be enabled")
        if fingerprint_size <= 0:
            raise ValueError("fingerprint_size must be positive")

        self.handle_errors = handle_errors
        self.missing_value_strategy = missing_value_strategy
        self.include_descriptors = include_descriptors
        self.fingerprint_type = fingerprint_type
        self.fingerprint_size = fingerprint_size
        self.fingerprint_radius = fingerprint_radius

        selected_names = self._resolve_descriptor_names(
            descriptor_names=descriptor_names,
            descriptor_set=descriptor_set,
        )
        self._descriptor_names = selected_names if include_descriptors else []

        self._imputation_values: np.ndarray | None = None
        self._normalization_means: np.ndarray | None = None
        self._normalization_stds: np.ndarray | None = None

    @staticmethod
    def available_descriptor_names() -> list[str]:
        """Return all public RDKit descriptor names."""
        return list(_ALL_DESCRIPTOR_NAMES)

    @staticmethod
    def descriptor_subsets() -> dict[str, list[str]]:
        """Return curated descriptor subsets used for baseline experiments."""
        return {name: list(values) for name, values in _SUBSET_MAP.items()}

    @property
    def descriptor_names(self) -> list[str]:
        """Return the configured descriptor names."""
        return list(self._descriptor_names)

    @property
    def feature_names(self) -> list[str]:
        """Return descriptor and fingerprint feature names."""
        names = list(self._descriptor_names)
        if self.fingerprint_type is not None:
            prefix = self.fingerprint_type
            size = self._fingerprint_length()
            names.extend(f"{prefix}_bit_{index}" for index in range(size))
        return names

    @property
    def num_descriptors(self) -> int:
        """Return the number of configured descriptors."""
        return len(self._descriptor_names)

    @property
    def num_features(self) -> int:
        """Return the full feature count."""
        return len(self.feature_names)

    def extract(
        self,
        molecule: Chem.Mol | None,
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract features for a single molecule."""
        feature_row = self._extract_single(molecule)
        matrix = feature_row.reshape(1, -1)
        prepared = self._prepare_matrix(matrix, normalize=normalize)
        return prepared[0]

    def extract_batch(
        self,
        molecules: Sequence[Chem.Mol | None],
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract features for a sequence of RDKit molecules."""
        return self._prepare_matrix(self._extract_matrix(molecules), normalize=normalize)

    def extract_from_smiles(
        self,
        smiles_list: Sequence[str],
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        """Extract features directly from a sequence of SMILES strings."""
        molecules = [self._load_smiles_or_none(smiles) for smiles in smiles_list]
        return self.extract_batch(molecules, normalize=normalize)

    def fit(
        self,
        molecules: Sequence[Chem.Mol | str | None],
        *,
        normalize: bool = True,
    ) -> DescriptorExtractor:
        """Fit missing-value and normalization statistics on a training batch."""
        raw_matrix = self._extract_matrix(self._coerce_iterable(molecules))
        self._imputation_values = self._compute_imputation_values(raw_matrix)

        prepared = self._apply_missing_values(raw_matrix)
        if normalize:
            self._normalization_means = np.nanmean(prepared, axis=0)
            self._normalization_stds = np.nanstd(prepared, axis=0)
        else:
            self._normalization_means = None
            self._normalization_stds = None

        return self

    def transform(
        self,
        molecules: Sequence[Chem.Mol | str | None],
        normalize: bool = False,
    ) -> np.ndarray:
        """Transform molecules or SMILES into a feature matrix."""
        raw_matrix = self._extract_matrix(self._coerce_iterable(molecules))
        return self._prepare_matrix(raw_matrix, normalize=normalize)

    def fit_transform(
        self,
        molecules: Sequence[Chem.Mol | str | None],
        normalize: bool = True,
    ) -> np.ndarray:
        """Fit training statistics and return the transformed feature matrix."""
        self.fit(molecules, normalize=normalize)
        return self.transform(molecules, normalize=normalize)

    def to_dataframe(
        self,
        molecules: Sequence[Chem.Mol | str | None],
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Return extracted features as a pandas DataFrame."""
        return pd.DataFrame(
            self.transform(molecules, normalize=normalize),
            columns=self.feature_names,
        )

    def get_valid_descriptor_mask(self, X: np.ndarray) -> np.ndarray:
        """Return a mask for descriptor columns without missing values."""
        descriptor_matrix = np.asarray(X, dtype=float)[:, : self.num_descriptors]
        return ~np.any(np.isnan(descriptor_matrix), axis=0)

    def _resolve_descriptor_names(
        self,
        descriptor_names: Sequence[str] | None,
        descriptor_set: DescriptorSet,
    ) -> list[str]:
        names = _SUBSET_MAP[descriptor_set] if descriptor_names is None else list(descriptor_names)

        invalid_names = sorted(set(names).difference(_DESCRIPTOR_FUNCTIONS))
        if invalid_names:
            raise ValueError(f"Unknown descriptor names: {invalid_names}")

        return names

    def _coerce_iterable(
        self,
        molecules: Sequence[Chem.Mol | str | None],
    ) -> list[Chem.Mol | None]:
        coerced: list[Chem.Mol | None] = []
        for entry in molecules:
            if isinstance(entry, str):
                coerced.append(self._load_smiles_or_none(entry))
            else:
                coerced.append(entry)
        return coerced

    def _load_smiles_or_none(self, smiles: str) -> Chem.Mol | None:
        try:
            return load_smiles(smiles)
        except InvalidSMILESError:
            if self.handle_errors == "raise":
                raise
            logger.warning("Failed to parse SMILES %r", smiles)
            return None

    def _extract_matrix(self, molecules: Iterable[Chem.Mol | None]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for molecule in molecules:
            if molecule is None and self.handle_errors == "skip":
                continue
            rows.append(self._extract_single(molecule))

        if not rows:
            return np.empty((0, self.num_features), dtype=float)

        return np.vstack(rows)

    def _extract_single(self, molecule: Chem.Mol | None) -> np.ndarray:
        if molecule is None:
            if self.handle_errors == "raise":
                raise InvalidSMILESError("Cannot extract descriptors from an invalid molecule")
            return np.full(self.num_features, np.nan, dtype=float)

        feature_blocks: list[np.ndarray] = []
        if self.include_descriptors:
            feature_blocks.append(self._calculate_descriptors(molecule))
        if self.fingerprint_type is not None:
            feature_blocks.append(self._calculate_fingerprint(molecule))

        if not feature_blocks:
            return np.empty(0, dtype=float)

        return np.concatenate(feature_blocks).astype(float, copy=False)

    def _calculate_descriptors(self, molecule: Chem.Mol) -> np.ndarray:
        values = np.full(len(self._descriptor_names), np.nan, dtype=float)

        for index, name in enumerate(self._descriptor_names):
            descriptor_fn = _DESCRIPTOR_FUNCTIONS[name]
            try:
                value = descriptor_fn(molecule)
                numeric_value = float(value)
                if np.isfinite(numeric_value):
                    values[index] = numeric_value
            except Exception:
                if self.handle_errors == "raise":
                    raise
                logger.debug("Descriptor %s failed for molecule", name, exc_info=True)

        return values

    def _calculate_fingerprint(self, molecule: Chem.Mol) -> np.ndarray:
        if self.fingerprint_type == "morgan":
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(
                molecule,
                radius=self.fingerprint_radius,
                nBits=self.fingerprint_size,
            )
        elif self.fingerprint_type == "rdkit":
            fingerprint = Chem.RDKFingerprint(molecule, fpSize=self.fingerprint_size)
        elif self.fingerprint_type == "maccs":
            fingerprint = MACCSkeys.GenMACCSKeys(molecule)
        else:
            raise ValueError(f"Unsupported fingerprint_type: {self.fingerprint_type}")

        array = np.zeros((self._fingerprint_length(),), dtype=float)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _fingerprint_length(self) -> int:
        return 167 if self.fingerprint_type == "maccs" else self.fingerprint_size

    def _prepare_matrix(self, matrix: np.ndarray, *, normalize: bool) -> np.ndarray:
        prepared = self._apply_missing_values(matrix)
        if normalize:
            if self._normalization_means is None or self._normalization_stds is None:
                raise RuntimeError(
                    "Cannot normalize features before calling fit() or fit_transform()."
                )
            prepared = self._apply_normalization(prepared)
        return prepared

    def _compute_imputation_values(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.empty((self.num_features,), dtype=float)

        if self.missing_value_strategy == "zero":
            return np.zeros(matrix.shape[1], dtype=float)
        if self.missing_value_strategy == "nan":
            return np.full(matrix.shape[1], np.nan, dtype=float)

        values = np.nanmean(matrix, axis=0)
        return np.where(np.isnan(values), 0.0, values)

    def _apply_missing_values(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.size == 0:
            return matrix.copy()

        imputation_values = (
            self._imputation_values
            if self._imputation_values is not None
            else self._compute_imputation_values(matrix)
        )

        prepared = np.where(np.isinf(matrix), np.nan, matrix).copy()
        if self.missing_value_strategy == "nan":
            return prepared

        if imputation_values.shape[0] != prepared.shape[1]:
            raise RuntimeError("Feature dimension does not match fitted imputation statistics")

        nan_rows, nan_cols = np.where(np.isnan(prepared))
        if len(nan_rows) > 0:
            prepared[nan_rows, nan_cols] = imputation_values[nan_cols]
        return prepared

    def _apply_normalization(self, matrix: np.ndarray) -> np.ndarray:
        std = np.where(self._normalization_stds == 0.0, 1.0, self._normalization_stds)
        return (matrix - self._normalization_means) / std


MolecularDescriptorExtractor = DescriptorExtractor

__all__ = [
    "DescriptorExtractor",
    "MolecularDescriptorExtractor",
    "FingerprintType",
    "HandleErrors",
    "MissingValueStrategy",
]
