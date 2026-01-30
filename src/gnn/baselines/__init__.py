"""Traditional ML baseline models for molecular property prediction.

This module provides baseline models (Random Forest, XGBoost) that use
RDKit molecular descriptors for PFAS property prediction.

Example:
    >>> from gnn.data import PFASBenchDataModule
    >>> from gnn.baselines import (
    ...     MolecularDescriptorExtractor,
    ...     RandomForestBaseline,
    ...     extract_baseline_data,
    ... )
    >>> from gnn.evaluation import ModelComparison
    >>>
    >>> # Setup data (same splits as GNN)
    >>> dm = PFASBenchDataModule(root="data/", split="scaffold", seed=42)
    >>> dm.setup()
    >>>
    >>> # Extract features
    >>> data = extract_baseline_data(dm)
    >>>
    >>> # Train model
    >>> model = RandomForestBaseline(n_estimators=100)
    >>> model.fit(data.X_train, data.y_train)
    >>>
    >>> # Evaluate
    >>> predictions = model.predict(data.X_test)
    >>> comparison = ModelComparison(property_names=["logS", "logP", "pKa"])
    >>> comparison.add_result("RandomForest", predictions, data.y_test)
    >>> print(comparison.to_table())
"""

from gnn.baselines.data_utils import BaselineDataset, extract_baseline_data
from gnn.baselines.descriptors import MolecularDescriptorExtractor
from gnn.baselines.models import (
    BaselineModel,
    MultiOutputBaselineModel,
    RandomForestBaseline,
    XGBoostBaseline,
)

__all__ = [
    # Descriptors
    "MolecularDescriptorExtractor",
    # Models
    "BaselineModel",
    "MultiOutputBaselineModel",
    "RandomForestBaseline",
    "XGBoostBaseline",
    # Data utilities
    "BaselineDataset",
    "extract_baseline_data",
]
