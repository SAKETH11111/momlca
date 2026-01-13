"""PFAS-family out-of-distribution data splitting utilities.

This module provides functions for splitting molecular datasets using
PFAS-family characteristics (chain length, headgroup type), enabling
evaluation of model generalization to unseen PFAS families.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from rdkit import Chem

from gnn.data.loaders import load_smiles

logger = logging.getLogger(__name__)

_HEADGROUP_SMARTS: dict[str, str] = {
    "carboxylate": "[CX3](=O)[OX2H1,OX1-]",  # -COOH or -COO-
    "sulfonate": "[SX4](=O)(=O)[OX2H1,OX1-]",  # -SO3H or -SO3-
    "phosphonate": "[PX4](=O)([OX2H1,OX1-])([OX2H1,OX1-])",  # phosphonic acid
    "alcohol": "[CX4][OX2H1]",  # -CH2-OH (not connected to C=O)
}

_HEADGROUP_PATTERNS: dict[str, Chem.Mol] = {
    name: pattern
    for name, smarts in _HEADGROUP_SMARTS.items()
    if (pattern := Chem.MolFromSmarts(smarts)) is not None
}


def get_chain_length(mol: Chem.Mol) -> str:
    """Extract perfluoroalkyl chain length from PFAS molecule.

    Counts the total number of carbon atoms in the molecule to determine
    the chain length category. For PFAS molecules, this corresponds to
    the perfluoroalkyl chain length (e.g., C4, C8).

    Args:
        mol: RDKit Mol object

    Returns:
        Chain length label (e.g., "C2", "C4", "C8") or "unknown" if no carbons
    """
    carbon_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)

    if carbon_count == 0:
        return "unknown"

    return f"C{carbon_count}"


def get_headgroup(mol: Chem.Mol) -> str:
    """Extract functional headgroup from PFAS molecule.

    Identifies the terminal functional group of a PFAS molecule using
    SMARTS pattern matching. Common PFAS headgroups include carboxylate
    (-COOH), sulfonate (-SO3H), phosphonate, and alcohol (-OH).

    Args:
        mol: RDKit Mol object

    Returns:
        Headgroup label: "carboxylate", "sulfonate", "phosphonate",
        "alcohol", or "other"
    """
    for headgroup, pattern in _HEADGROUP_PATTERNS.items():
        if mol.HasSubstructMatch(pattern):
            return headgroup

    return "other"


@runtime_checkable
class HasSmiles(Protocol):
    """Protocol for objects with a smiles attribute."""

    smiles: str


def _compute_split_stats(
    labels: list[str],
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> dict[str, dict[str, int]]:
    """Compute statistics for each split.

    Args:
        labels: List of family labels for all molecules
        train_indices: Indices of training molecules
        val_indices: Indices of validation molecules
        test_indices: Indices of test molecules

    Returns:
        Dictionary with counts per family for each split
    """
    stats: dict[str, dict[str, int]] = {"train": {}, "val": {}, "test": {}}

    for name, indices in [
        ("train", train_indices),
        ("val", val_indices),
        ("test", test_indices),
    ]:
        for idx in indices:
            label = labels[idx]
            stats[name][label] = stats[name].get(label, 0) + 1

    return stats


def _format_family_counts(counts: dict[str, int]) -> str:
    if not counts:
        return "{}"
    return ", ".join(f"{label}={count}" for label, count in sorted(counts.items()))


def pfas_ood_split(
    data: Sequence[str] | Sequence[HasSmiles],
    holdout: str = "chain_length",
    holdout_values: list[str] | None = None,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, dict[str, int]]]:
    """Split dataset with PFAS family out-of-distribution test set.

    Places molecules matching the holdout criteria in the test set,
    ensuring the model is evaluated on unseen PFAS families. Remaining
    molecules are randomly split into train/val sets.

    Args:
        data: Either a list of SMILES strings, or a sequence of objects
            with a `smiles` attribute (e.g., PyG Data objects or a Dataset).
        holdout: Family type to hold out ("chain_length" or "headgroup")
        holdout_values: Family values to place in test set (e.g., ["C2", "C3"])
        seed: Random seed for train/val split
        train_frac: Fraction of non-holdout data for training
        val_frac: Fraction of non-holdout data for validation

    Returns:
        Tuple of (train_indices, val_indices, test_indices, stats_dict)

    Raises:
        ValueError: If holdout type is not recognized
        ValueError: If train/val fractions are invalid
        TypeError: If input does not provide SMILES strings
        InvalidSMILESError: If an input SMILES is invalid

    Example:
        >>> smiles = ["C(=O)(C(F)(F)F)O", "C(=O)(C(C(F)(F)F)(F)F)O"]
        >>> train, val, test, stats = pfas_ood_split(smiles, holdout="chain_length", holdout_values=["C2"])
        >>> len(test) > 0  # C2 molecules in test
        True
    """
    if holdout_values is None:
        holdout_values = []

    if holdout not in ("chain_length", "headgroup"):
        raise ValueError(f"Unknown holdout type: {holdout}. Must be 'chain_length' or 'headgroup'")

    for name, frac in (("train_frac", train_frac), ("val_frac", val_frac)):
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1, got {frac}")
    if abs(train_frac + val_frac - 1.0) > 1e-6:
        raise ValueError(f"train_frac + val_frac must equal 1.0, got {train_frac + val_frac:.6f}")

    n_total = len(data)

    # Extract SMILES and convert to molecules
    mols: list[Chem.Mol] = []
    for i in range(n_total):
        item = data[i]
        if isinstance(item, str):
            smiles = item
        elif isinstance(item, HasSmiles):
            smiles = str(item.smiles)
        else:
            raise TypeError(f"Expected string or object with 'smiles' attribute, got {type(item)}")

        smiles = str(smiles).strip()
        if not smiles:
            raise ValueError(f"Empty SMILES at index {i}")

        mol = load_smiles(smiles)
        mols.append(mol)

    # Get family labels for all molecules
    if holdout == "chain_length":
        labels = [get_chain_length(mol) for mol in mols]
    else:  # headgroup
        labels = [get_headgroup(mol) for mol in mols]

    # Split into holdout (test) and non-holdout (train/val)
    test_indices = [i for i, label in enumerate(labels) if label in holdout_values]
    remaining_indices = [i for i, label in enumerate(labels) if label not in holdout_values]

    # Random split of remaining into train/val
    rng = np.random.default_rng(seed)
    remaining_array = np.array(remaining_indices)
    rng.shuffle(remaining_array)
    remaining_indices = remaining_array.tolist()

    # Calculate split point
    if len(remaining_indices) > 0:
        n_train = int(len(remaining_indices) * train_frac / (train_frac + val_frac))
        train_indices = remaining_indices[:n_train]
        val_indices = remaining_indices[n_train:]
    else:
        train_indices = []
        val_indices = []

    # Compute statistics
    stats = _compute_split_stats(labels, train_indices, val_indices, test_indices)

    train_set = set(train_indices)
    val_set = set(val_indices)
    test_set = set(test_indices)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Split overlap detected: an index appears in multiple splits")

    all_assigned = train_set | val_set | test_set
    expected = set(range(n_total))
    if all_assigned != expected:
        missing = sorted(expected - all_assigned)
        extra = sorted(all_assigned - expected)
        raise ValueError(f"Split coverage invalid (missing={missing}, extra={extra})")

    # Log statistics
    logger.info("OOD Split (%s=%s):", holdout, holdout_values)
    logger.info(
        "  Train: %d molecules (%s)",
        len(train_indices),
        _format_family_counts(stats["train"]),
    )
    logger.info(
        "  Val: %d molecules (%s)",
        len(val_indices),
        _format_family_counts(stats["val"]),
    )
    logger.info(
        "  Test (OOD): %d molecules (%s)",
        len(test_indices),
        _format_family_counts(stats["test"]),
    )

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
        stats,
    )
