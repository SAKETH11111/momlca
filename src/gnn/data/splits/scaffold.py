"""Scaffold-based data splitting utilities.

This module provides functions for splitting molecular datasets using
Bemis-Murcko scaffolds, ensuring that molecules with the same scaffold
are kept together in the same split. This tests model generalization
to structurally novel compounds.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from gnn.data.loaders import load_smiles

logger = logging.getLogger(__name__)


def get_scaffold(mol: Chem.Mol) -> str:
    """Extract Bemis-Murcko scaffold from molecule.

    The Bemis-Murcko scaffold is the core ring structure of a molecule
    with all side chains removed. This is useful for grouping molecules
    by structural similarity.

    Args:
        mol: RDKit Mol object

    Returns:
        Canonical SMILES of the scaffold, or empty string if the molecule
        has no rings (acyclic molecules have no scaffold).
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
        # Empty scaffold (no rings) returns empty string
        if scaffold_smiles == "" or scaffold.GetNumAtoms() == 0:
            return ""
        return scaffold_smiles
    except Exception:
        # Some molecules may fail scaffold extraction
        logger.warning("Failed to extract scaffold for molecule", exc_info=True)
        return ""


def group_by_scaffold(mols: list[Chem.Mol]) -> dict[str, list[int]]:
    """Group molecules by their Bemis-Murcko scaffold.

    Molecules with the same scaffold structure are grouped together,
    which is useful for creating scaffold-based train/val/test splits.

    Args:
        mols: List of RDKit Mol objects

    Returns:
        Dictionary mapping scaffold SMILES to list of molecule indices.
        Molecules without scaffolds (acyclic) are grouped under empty string.
    """
    scaffolds: dict[str, list[int]] = {}
    for idx, mol in enumerate(mols):
        scaffold = get_scaffold(mol)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(idx)
    return scaffolds


@runtime_checkable
class HasSmiles(Protocol):
    """Protocol for objects with a smiles attribute."""

    smiles: str


def scaffold_split(
    data: Sequence[str] | Sequence[HasSmiles],
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    train: float | None = None,
    val: float | None = None,
    test: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset by Bemis-Murcko scaffold.

    Molecules sharing the same scaffold are kept together in the same split,
    which tests model generalization to structurally novel compounds.
    Scaffolds are shuffled deterministically before assignment to achieve
    approximate target proportions while respecting scaffold boundaries.

    Args:
        data: Either a list of SMILES strings, or a sequence of objects
            with a `smiles` attribute (e.g., PyG Data objects or a Dataset).
        seed: Random seed for reproducibility.
        train_frac: Fraction of data for training (default 0.8).
        val_frac: Fraction of data for validation (default 0.1).
        test_frac: Fraction of data for testing (default 0.1).
        train: Alias for train fraction (matches story AC keyword).
        val: Alias for val fraction (matches story AC keyword).
        test: Alias for test fraction (matches story AC keyword).

    Returns:
        Tuple of (train_indices, val_indices, test_indices) as numpy arrays.

    Raises:
        ValueError: If fractions don't sum to 1.0 (within tolerance).

    Example:
        >>> smiles = ["c1ccccc1", "Cc1ccccc1", "c1ccc2ccccc2c1", "CCC"]
        >>> train, val, test = scaffold_split(smiles, seed=42)
        >>> len(train) + len(val) + len(test) == 4
        True
    """
    if train is not None:
        if not np.isclose(train, train_frac):
            raise ValueError("Conflicting values: provide either train or train_frac")
        train_frac = float(train)
    if val is not None:
        if not np.isclose(val, val_frac):
            raise ValueError("Conflicting values: provide either val or val_frac")
        val_frac = float(val)
    if test is not None:
        if not np.isclose(test, test_frac):
            raise ValueError("Conflicting values: provide either test or test_frac")
        test_frac = float(test)

    # Validate fractions
    for name, frac in (
        ("train_frac", train_frac),
        ("val_frac", val_frac),
        ("test_frac", test_frac),
    ):
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"{name} must be between 0 and 1, got {frac}")
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-6:
        raise ValueError(
            f"train_frac + val_frac + test_frac must equal 1.0, "
            f"got {train_frac + val_frac + test_frac:.6f}"
        )

    # Extract SMILES and convert to molecules
    n_total = len(data)
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

    # Group molecules by scaffold
    scaffold_groups = group_by_scaffold(mols)
    logger.info("Found %d unique scaffolds for %d molecules", len(scaffold_groups), n_total)

    # Shuffle scaffolds deterministically
    rng = np.random.default_rng(seed)
    scaffold_list = list(scaffold_groups.keys())
    rng.shuffle(scaffold_list)

    # Calculate target sizes
    n_train = int(n_total * train_frac)
    n_val = int(n_total * val_frac)

    # Assign scaffolds to splits
    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    for scaffold in scaffold_list:
        indices = scaffold_groups[scaffold]
        if len(train_indices) < n_train:
            train_indices.extend(indices)
        elif len(train_indices) + len(val_indices) < n_train + n_val:
            val_indices.extend(indices)
        else:
            test_indices.extend(indices)

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_indices),
        len(val_indices),
        len(test_indices),
    )

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

    return (
        np.array(train_indices, dtype=np.int64),
        np.array(val_indices, dtype=np.int64),
        np.array(test_indices, dtype=np.int64),
    )
