"""Data utilities for ML baselines.

Provides functions to extract features and labels from PFASBenchDataModule
for use with scikit-learn style models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from gnn.baselines.descriptors import MolecularDescriptorExtractor

if TYPE_CHECKING:
    from gnn.data.datamodules import PFASBenchDataModule

logger = logging.getLogger(__name__)


@dataclass
class BaselineDataset:
    """Container for baseline training/evaluation data.

    Attributes:
        X_train: Training features (n_train, n_features)
        y_train: Training labels (n_train, n_properties)
        X_val: Validation features (n_val, n_features)
        y_val: Validation labels (n_val, n_properties)
        X_test: Test features (n_test, n_features)
        y_test: Test labels (n_test, n_properties)
        feature_names: Names of features (descriptor names)
        property_names: Names of target properties
        smiles_train: Training SMILES strings
        smiles_val: Validation SMILES strings
        smiles_test: Test SMILES strings
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    property_names: list[str]
    smiles_train: list[str]
    smiles_val: list[str]
    smiles_test: list[str]


def extract_baseline_data(
    datamodule: PFASBenchDataModule,
    extractor: MolecularDescriptorExtractor | None = None,
    normalize_features: bool = True,
) -> BaselineDataset:
    """Extract features and labels from PFASBenchDataModule.

    Uses the exact same train/val/test splits as the GNN models.

    Args:
        datamodule: Initialized and setup PFASBenchDataModule
        extractor: Descriptor extractor (creates default if None)
        normalize_features: Whether to z-score normalize features

    Returns:
        BaselineDataset with extracted features and labels

    Raises:
        RuntimeError: If datamodule is not setup

    Example:
        >>> from gnn.data import PFASBenchDataModule
        >>> from gnn.baselines import extract_baseline_data
        >>> dm = PFASBenchDataModule(root="data/", split="scaffold", seed=42)
        >>> dm.setup()
        >>> data = extract_baseline_data(dm)
        >>> data.X_train.shape
        (112, 208)
    """
    # Ensure datamodule is setup
    if datamodule.dataset is None:
        raise RuntimeError("DataModule not setup. Call datamodule.setup() first.")

    if datamodule.train_idx is None:
        raise RuntimeError("Split indices not computed. Call datamodule.setup() first.")

    # Create extractor if needed
    if extractor is None:
        extractor = MolecularDescriptorExtractor()

    # Get property names from dataset
    property_names = datamodule.dataset.property_names

    # Extract SMILES for each split
    smiles_train = [datamodule.dataset.get_smiles(int(i)) for i in datamodule.train_idx]
    smiles_val = [datamodule.dataset.get_smiles(int(i)) for i in datamodule.val_idx]
    smiles_test = [datamodule.dataset.get_smiles(int(i)) for i in datamodule.test_idx]

    logger.info(
        "Extracting descriptors: train=%d, val=%d, test=%d",
        len(smiles_train),
        len(smiles_val),
        len(smiles_test),
    )

    # Extract features (fit on train, transform on all)
    X_train = extractor.fit_transform(smiles_train, normalize=normalize_features)
    X_val = extractor.transform(smiles_val, normalize=normalize_features)
    X_test = extractor.transform(smiles_test, normalize=normalize_features)

    # Extract labels from dataset
    def get_labels(indices: np.ndarray) -> np.ndarray:
        labels = []
        for idx in indices:
            data = datamodule.dataset[int(idx)]
            # y is shape (1, num_properties), squeeze to (num_properties,)
            y = data.y.squeeze(0).numpy()
            labels.append(y)
        return np.array(labels)

    y_train = get_labels(datamodule.train_idx)
    y_val = get_labels(datamodule.val_idx)
    y_test = get_labels(datamodule.test_idx)

    logger.info(
        "Extracted data shapes: X_train=%s, y_train=%s, X_test=%s, y_test=%s",
        X_train.shape,
        y_train.shape,
        X_test.shape,
        y_test.shape,
    )

    return BaselineDataset(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=extractor.descriptor_names,
        property_names=property_names,
        smiles_train=smiles_train,
        smiles_val=smiles_val,
        smiles_test=smiles_test,
    )
