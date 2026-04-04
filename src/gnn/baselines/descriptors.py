"""Descriptor and fingerprint extraction for tabular baseline models."""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, cast

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
HandleNan = Literal["drop", "impute_mean", "impute_median", "impute_zero"]
MissingValueStrategy = Literal["nan", "mean", "median", "zero"]

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

THREE_D_DESCRIPTOR_NAMES = {
    "Asphericity",
    "Eccentricity",
    "InertialShapeFactor",
    "NPR1",
    "NPR2",
    "PBF",
    "PMI1",
    "PMI2",
    "PMI3",
    "RadiusOfGyration",
    "SpherocityIndex",
}

_DESCRIPTOR_FUNCTIONS = dict(Descriptors.descList)
_ALL_DESCRIPTOR_NAMES = [name for name, _ in Descriptors.descList]
_TWO_D_DESCRIPTOR_NAMES = [
    name for name in _ALL_DESCRIPTOR_NAMES if name not in THREE_D_DESCRIPTOR_NAMES
]


def _descriptor_subset_map() -> dict[DescriptorSet, list[str]]:
    return {
        "all": list(_ALL_DESCRIPTOR_NAMES),
        "2d": list(_TWO_D_DESCRIPTOR_NAMES),
        "physicochemical": list(PHYSICOCHEMICAL_DESCRIPTOR_NAMES),
        "topological": list(TOPOLOGICAL_DESCRIPTOR_NAMES),
    }


def export_pfasbench_descriptors(
    input_csv: str | Path,
    output_path: str | Path,
    *,
    extractor: DescriptorExtractor | None = None,
    normalize: bool = False,
) -> Path:
    """Extract PFASBench descriptors and persist them as a parquet dataset."""
    extractor = extractor or DescriptorExtractor()
    input_path = Path(input_csv)
    output = Path(output_path)

    frame = pd.read_csv(input_path)
    if "smiles" not in frame.columns:
        raise ValueError(f"Expected a 'smiles' column in {input_path}")

    smiles_values = frame["smiles"].fillna("").tolist()
    kept_indices: list[int] = []
    molecules: list[Chem.Mol | None] = []
    invalid_rows: list[int] = []

    for source_index, smiles in enumerate(smiles_values):
        molecule = extractor._load_smiles_or_none(smiles)
        if molecule is None and extractor.handle_errors == "skip":
            continue

        kept_indices.append(source_index)
        molecules.append(molecule)
        if molecule is None:
            invalid_rows.append(len(molecules) - 1)

    features = extractor.extract_batch(molecules, normalize=normalize)
    if invalid_rows:
        features.iloc[invalid_rows, :] = np.nan
        logger.warning(
            "Descriptor export preserved %d invalid SMILES rows as NaN features",
            len(invalid_rows),
        )

    if len(kept_indices) != len(frame):
        logger.warning(
            "Descriptor export skipped %d invalid SMILES rows due to handle_errors='skip'",
            len(frame) - len(kept_indices),
        )

    export_frame = pd.concat(
        [frame.iloc[kept_indices].reset_index(drop=True), features.reset_index(drop=True)],
        axis=1,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        export_frame.to_parquet(output, index=False)
    except ImportError as exc:
        raise ImportError(
            "Writing parquet files requires an optional parquet engine such as "
            "'pyarrow' or 'fastparquet'."
        ) from exc

    logger.info("Saved descriptor export for %d molecules to %s", len(export_frame), output)
    return output


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
        handle_nan: HandleNan = "impute_mean",
        missing_value_strategy: MissingValueStrategy | None = None,
    ) -> None:
        if handle_errors not in {"nan", "raise", "skip"}:
            raise ValueError("handle_errors must be one of: 'nan', 'raise', 'skip'")
        if handle_nan not in {"drop", "impute_mean", "impute_median", "impute_zero"}:
            raise ValueError(
                "handle_nan must be one of: 'drop', 'impute_mean', 'impute_median', 'impute_zero'"
            )
        if missing_value_strategy is not None and missing_value_strategy not in {
            "nan",
            "mean",
            "median",
            "zero",
        }:
            raise ValueError(
                "missing_value_strategy must be one of: 'nan', 'mean', 'median', 'zero'"
            )
        if not include_descriptors and fingerprint_type is None:
            raise ValueError("At least one feature source must be enabled")
        if fingerprint_size <= 0:
            raise ValueError("fingerprint_size must be positive")

        self.handle_errors = handle_errors
        self.handle_nan = handle_nan
        self.include_descriptors = include_descriptors
        self.fingerprint_type = fingerprint_type
        self.fingerprint_size = fingerprint_size
        self.fingerprint_radius = fingerprint_radius
        self._missing_value_mode = self._resolve_missing_value_mode(
            handle_nan=handle_nan,
            missing_value_strategy=missing_value_strategy,
        )

        selected_names = self._resolve_descriptor_names(
            descriptor_names=descriptor_names,
            descriptor_set=descriptor_set,
        )
        self._descriptor_names = selected_names if include_descriptors else []

        self._imputation_values: np.ndarray | None = None
        self._normalization_means: np.ndarray | None = None
        self._normalization_stds: np.ndarray | None = None
        self._drop_mask: np.ndarray | None = None
        self._last_descriptor_failures: dict[str, int] = {}

    @staticmethod
    def available_descriptor_names() -> list[str]:
        """Return all public RDKit descriptor names."""
        return list(_ALL_DESCRIPTOR_NAMES)

    @staticmethod
    def descriptor_subsets() -> dict[str, list[str]]:
        """Return curated descriptor subsets used for baseline experiments."""
        return cast(dict[str, list[str]], _descriptor_subset_map())

    @property
    def descriptor_names(self) -> list[str]:
        """Return the configured descriptor names."""
        return list(self._descriptor_names)

    @property
    def feature_names(self) -> list[str]:
        """Return descriptor and fingerprint feature names."""
        return self._feature_names_for_mask(self._drop_mask)

    @property
    def num_descriptors(self) -> int:
        """Return the number of configured descriptors."""
        return len(self._descriptor_names)

    @property
    def num_features(self) -> int:
        """Return the full feature count before any NaN-drop filtering."""
        return len(self._full_feature_names())

    def extract(
        self,
        molecule: Chem.Mol | None,
        *,
        normalize: bool = False,
    ) -> dict[str, float]:
        """Extract features for a single molecule."""
        matrix, feature_names = self._prepare_matrix_with_feature_names(
            self._extract_single(molecule).reshape(1, -1),
            normalize=normalize,
        )
        if matrix.shape[0] == 0:
            return {}
        return dict(zip(feature_names, matrix[0].tolist(), strict=False))

    def extract_batch(
        self,
        molecules: Sequence[Chem.Mol | None],
        *,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Extract features for a sequence of RDKit molecules."""
        matrix, feature_names = self._prepare_matrix_with_feature_names(
            self._extract_matrix(molecules),
            normalize=normalize,
        )
        return pd.DataFrame(matrix, columns=feature_names)

    def extract_from_smiles(
        self,
        smiles: str | Sequence[str],
        *,
        normalize: bool = False,
    ) -> pd.DataFrame:
        """Extract features directly from one or more SMILES strings."""
        smiles_list = [smiles] if isinstance(smiles, str) else list(smiles)
        molecules = [self._load_smiles_or_none(item) for item in smiles_list]
        return self.extract_batch(molecules, normalize=normalize)

    def fit(
        self,
        molecules: Sequence[Chem.Mol | str | None],
        *,
        normalize: bool = True,
    ) -> DescriptorExtractor:
        """Fit missing-value and normalization statistics on a training batch."""
        raw_matrix = self._extract_matrix(self._coerce_iterable(molecules))
        prepared = np.where(np.isinf(raw_matrix), np.nan, raw_matrix).copy()

        if self._missing_value_mode == "drop":
            self._drop_mask = ~np.any(np.isnan(prepared), axis=0)
            prepared = prepared[:, self._drop_mask]
            self._imputation_values = None
        else:
            self._drop_mask = None
            self._imputation_values = self._compute_imputation_values(prepared)
            prepared = self._apply_missing_values(prepared)

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
        matrix, feature_names = self._prepare_matrix_with_feature_names(
            self._extract_matrix(self._coerce_iterable(molecules)),
            normalize=normalize,
        )
        return pd.DataFrame(matrix, columns=feature_names)

    def get_feature_names(self) -> list[str]:
        """Return feature names for the current extractor configuration."""
        return self.feature_names

    def get_valid_descriptor_mask(self, X: np.ndarray) -> np.ndarray:
        """Return a mask for descriptor columns without missing values."""
        if self._drop_mask is not None:
            return self._drop_mask[: self.num_descriptors].copy()

        descriptor_matrix = np.asarray(X, dtype=float)[:, : self.num_descriptors]
        return ~np.any(np.isnan(descriptor_matrix), axis=0)

    def compute_fingerprints(
        self,
        molecule: Chem.Mol,
        *,
        fp_type: FingerprintType = None,
        radius: int | None = None,
        nbits: int | None = None,
    ) -> np.ndarray:
        """Compute a molecular fingerprint as a numpy vector."""
        if molecule is None:
            raise ValueError("Cannot compute fingerprints for an invalid molecule")

        fingerprint_type = fp_type or self.fingerprint_type or "morgan"
        fingerprint_radius = radius if radius is not None else self.fingerprint_radius
        fingerprint_size = nbits if nbits is not None else self.fingerprint_size

        if fingerprint_type == "morgan":
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(  # type: ignore[attr-defined]
                molecule,
                radius=fingerprint_radius,
                nBits=fingerprint_size,
            )
            length = fingerprint_size
        elif fingerprint_type == "rdkit":
            fingerprint = Chem.RDKFingerprint(molecule, fpSize=fingerprint_size)
            length = fingerprint_size
        elif fingerprint_type == "maccs":
            fingerprint = MACCSkeys.GenMACCSKeys(molecule)  # type: ignore[attr-defined]
            length = 167
        else:
            raise ValueError(f"Unsupported fingerprint_type: {fingerprint_type}")

        array = np.zeros((length,), dtype=float)
        DataStructs.ConvertToNumpyArray(fingerprint, array)
        return array

    def _resolve_descriptor_names(
        self,
        descriptor_names: Sequence[str] | None,
        descriptor_set: DescriptorSet,
    ) -> list[str]:
        subset_map = _descriptor_subset_map()
        names = subset_map[descriptor_set] if descriptor_names is None else list(descriptor_names)

        invalid_names = sorted(set(names).difference(_DESCRIPTOR_FUNCTIONS))
        if invalid_names:
            raise ValueError(f"Unknown descriptor names: {invalid_names}")

        return names

    def _resolve_missing_value_mode(
        self,
        *,
        handle_nan: HandleNan,
        missing_value_strategy: MissingValueStrategy | None,
    ) -> str:
        if missing_value_strategy is None:
            return handle_nan

        legacy_map = {
            "nan": "nan",
            "mean": "impute_mean",
            "median": "impute_median",
            "zero": "impute_zero",
        }
        resolved = legacy_map[missing_value_strategy]
        if handle_nan != "impute_mean" and handle_nan != resolved:
            raise ValueError(
                "Specify either handle_nan or missing_value_strategy when choosing NaN handling"
            )
        return resolved

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
        failure_counts: Counter[str] = Counter()
        for molecule in molecules:
            if molecule is None and self.handle_errors == "skip":
                continue
            rows.append(self._extract_single(molecule, failure_counts=failure_counts))

        self._last_descriptor_failures = dict(failure_counts)
        if failure_counts:
            most_common = ", ".join(
                f"{name}={count}" for name, count in failure_counts.most_common(5)
            )
            logger.warning("Descriptor calculation failures observed: %s", most_common)

        if not rows:
            return np.empty((0, self.num_features), dtype=float)

        return np.vstack(rows)

    def _extract_single(
        self,
        molecule: Chem.Mol | None,
        *,
        failure_counts: Counter[str] | None = None,
    ) -> np.ndarray:
        if molecule is None:
            if self.handle_errors == "raise":
                raise InvalidSMILESError("Cannot extract descriptors from an invalid molecule")
            return np.full(self.num_features, np.nan, dtype=float)

        feature_blocks: list[np.ndarray] = []
        if self.include_descriptors:
            feature_blocks.append(
                self._calculate_descriptors(molecule, failure_counts=failure_counts)
            )
        if self.fingerprint_type is not None:
            feature_blocks.append(self.compute_fingerprints(molecule))

        if not feature_blocks:
            return np.empty(0, dtype=float)

        return np.concatenate(feature_blocks).astype(float, copy=False)

    def _calculate_descriptors(
        self,
        molecule: Chem.Mol,
        *,
        failure_counts: Counter[str] | None = None,
    ) -> np.ndarray:
        values = np.full(len(self._descriptor_names), np.nan, dtype=float)
        molecule_failures = 0

        for index, name in enumerate(self._descriptor_names):
            descriptor_fn = _DESCRIPTOR_FUNCTIONS[name]
            try:
                value = descriptor_fn(molecule)
                numeric_value = float(value)
                if np.isfinite(numeric_value):
                    values[index] = numeric_value
            except Exception:
                molecule_failures += 1
                if failure_counts is not None:
                    failure_counts[name] += 1
                if self.handle_errors == "raise":
                    raise
                logger.debug("Descriptor %s failed for molecule", name, exc_info=True)

        if molecule_failures >= max(3, len(self._descriptor_names) // 20):
            logger.warning(
                "Descriptor extraction produced %d failed values for one molecule",
                molecule_failures,
            )
        return values

    def _fingerprint_length(self) -> int:
        return 167 if self.fingerprint_type == "maccs" else self.fingerprint_size

    def _full_feature_names(self) -> list[str]:
        names = list(self._descriptor_names)
        if self.fingerprint_type is not None:
            prefix = self.fingerprint_type
            size = self._fingerprint_length()
            names.extend(f"{prefix}_bit_{index}" for index in range(size))
        return names

    def _feature_names_for_mask(self, mask: np.ndarray | None) -> list[str]:
        names = self._full_feature_names()
        if mask is None:
            return names
        return [name for keep, name in zip(mask, names, strict=False) if keep]

    def _prepare_matrix(self, matrix: np.ndarray, *, normalize: bool) -> np.ndarray:
        prepared, _ = self._prepare_matrix_with_feature_names(matrix, normalize=normalize)
        return prepared

    def _prepare_matrix_with_feature_names(
        self,
        matrix: np.ndarray,
        *,
        normalize: bool,
    ) -> tuple[np.ndarray, list[str]]:
        prepared = np.asarray(matrix, dtype=float)
        if prepared.size == 0:
            empty = prepared.copy()
            if empty.ndim == 1:
                empty = empty.reshape(0, 0)
            return empty, self.feature_names

        prepared = np.where(np.isinf(prepared), np.nan, prepared).copy()
        feature_names = self._full_feature_names()

        if self._missing_value_mode == "drop":
            drop_mask = (
                self._drop_mask
                if self._drop_mask is not None
                else ~np.any(np.isnan(prepared), axis=0)
            )
            self._drop_mask = drop_mask.copy()
            prepared = prepared[:, drop_mask]
            feature_names = self._feature_names_for_mask(drop_mask)
        else:
            prepared = self._apply_missing_values(prepared)

        if normalize:
            if self._normalization_means is None or self._normalization_stds is None:
                raise RuntimeError(
                    "Cannot normalize features before calling fit() or fit_transform()."
                )
            prepared = self._apply_normalization(prepared)

        return prepared, feature_names

    def _compute_imputation_values(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return np.empty((matrix.shape[1],), dtype=float)
        if self._missing_value_mode == "impute_zero":
            return np.zeros(matrix.shape[1], dtype=float)

        reducer = np.nanmedian if self._missing_value_mode == "impute_median" else np.nanmean
        values = reducer(matrix, axis=0)
        return np.where(np.isnan(values), 0.0, values)

    def _apply_missing_values(self, matrix: np.ndarray) -> np.ndarray:
        prepared = np.asarray(matrix, dtype=float).copy()
        if prepared.size == 0 or self._missing_value_mode == "nan":
            return prepared

        imputation_values = (
            self._imputation_values
            if self._imputation_values is not None
            else self._compute_imputation_values(prepared)
        )
        if imputation_values.shape[0] != prepared.shape[1]:
            raise RuntimeError("Feature dimension does not match fitted imputation statistics")

        nan_rows, nan_cols = np.where(np.isnan(prepared))
        if len(nan_rows) > 0:
            prepared[nan_rows, nan_cols] = imputation_values[nan_cols]
        return prepared

    def _apply_normalization(self, matrix: np.ndarray) -> np.ndarray:
        if self._normalization_means is None or self._normalization_stds is None:
            raise RuntimeError("Normalization statistics are not initialized")
        std = np.where(self._normalization_stds == 0.0, 1.0, self._normalization_stds)
        return (matrix - self._normalization_means) / std


MolecularDescriptorExtractor = DescriptorExtractor

__all__ = [
    "DescriptorExtractor",
    "DescriptorSet",
    "FingerprintType",
    "HandleErrors",
    "HandleNan",
    "MissingValueStrategy",
    "MolecularDescriptorExtractor",
    "export_pfasbench_descriptors",
]
