"""Data loading and processing utilities."""

from .datamodules import PFASBenchDataModule
from .datasets import PFASBenchDataset
from .loaders import load_mol, load_sdf, load_smiles
from .splits import (
    get_chain_length,
    get_headgroup,
    get_scaffold,
    group_by_scaffold,
    pfas_ood_split,
    scaffold_split,
)
from .transforms import (
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    MoleculeFeaturizer,
    get_atom_features,
    get_bond_features,
    get_edge_index,
    mol_to_pyg_data,
    mols_to_pyg_batch,
)
from .validation import DataValidator, ValidationResult, generate_report

__all__ = [
    # DataModules
    "PFASBenchDataModule",
    # Datasets
    "PFASBenchDataset",
    # Loaders
    "load_smiles",
    "load_sdf",
    "load_mol",
    # Scaffold splits
    "scaffold_split",
    "get_scaffold",
    "group_by_scaffold",
    # PFAS OOD splits
    "pfas_ood_split",
    "get_chain_length",
    "get_headgroup",
    # Featurizers
    "MoleculeFeaturizer",
    "get_atom_features",
    "get_bond_features",
    "get_edge_index",
    # PyG conversion
    "mol_to_pyg_data",
    "mols_to_pyg_batch",
    # Constants
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    # Validation
    "DataValidator",
    "ValidationResult",
    "generate_report",
]
