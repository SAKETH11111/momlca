"""3D conformer generation utilities for molecular structures.

This module provides functions to generate 3D conformers from RDKit Mol objects
using the ETKDG algorithm with optional MMFF force field optimization.
"""

import logging
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import AllChem as _AllChem

from gnn.exceptions import FeaturizationError

logger = logging.getLogger(__name__)
AllChem: Any = _AllChem


def generate_conformers(
    mol: Chem.Mol,
    num_conformers: int = 1,
    optimize: bool = True,
    random_seed: int = 42,
) -> list[torch.Tensor]:
    """Generate 3D conformers for a molecule.

    Uses the ETKDG (Experimental-Torsion Distance Geometry with basic Knowledge)
    algorithm for conformer generation. Optionally optimizes structures using
    the MMFF (Merck Molecular Force Field).

    Note:
        Hydrogens are added to a temporary copy of the molecule before conformer
        generation, which improves 3D embedding quality. The returned position
        tensors match the atom ordering and atom count of the input molecule
        (i.e., newly-added hydrogens are not included in the returned tensors).

    Args:
        mol: RDKit Mol object (2D or 3D input accepted)
        num_conformers: Number of conformers to generate (default: 1)
        optimize: Whether to optimize with MMFF force field (default: True)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        List of position tensors, each with shape (num_atoms, 3) and dtype float32,
        where num_atoms == mol.GetNumAtoms().

    Raises:
        FeaturizationError: If conformer generation fails (e.g., invalid molecule,
            embedding failure, single-atom molecule)

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")  # Ethanol
        >>> positions = generate_conformers(mol, num_conformers=3)
        >>> len(positions)
        3
        >>> positions[0].shape
        torch.Size([3, 3])  # 3 atoms from the input Mol
    """
    if mol is None:
        raise FeaturizationError("Cannot generate conformers: molecule is None")

    if num_conformers < 1:
        raise ValueError("num_conformers must be >= 1")

    # Check for single-atom molecules (cannot embed)
    if mol.GetNumAtoms() < 2:
        smiles = _get_smiles_safe(mol)
        raise FeaturizationError(f"Cannot generate conformers for single-atom molecule: {smiles}")

    num_input_atoms = mol.GetNumAtoms()

    # Add hydrogens (required for 3D embedding)
    mol_with_h = Chem.AddHs(mol)

    # ETKDG parameters for conformer generation
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed

    # Generate conformers
    conf_ids = AllChem.EmbedMultipleConfs(mol_with_h, numConfs=num_conformers, params=params)

    # Check for embedding failure
    if len(conf_ids) != num_conformers or any(conf_id < 0 for conf_id in conf_ids):
        smiles = _get_smiles_safe(mol)
        raise FeaturizationError(
            f"Failed to generate {num_conformers} conformer(s) for molecule: {smiles}"
        )

    # Optional MMFF optimization
    if optimize:
        _optimize_conformers_mmff(mol_with_h)

    # Extract positions as tensors
    positions = []
    for conf_id in conf_ids:
        conf = mol_with_h.GetConformer(conf_id)
        pos = conf.GetPositions()  # numpy array (num_atoms, 3)
        positions.append(torch.tensor(pos[:num_input_atoms], dtype=torch.float32))

    return positions


def get_positions(
    mol: Chem.Mol,
    optimize: bool = True,
    random_seed: int = 42,
) -> torch.Tensor:
    """Get 3D positions for a single conformer (convenience function).

    This is a simplified wrapper around generate_conformers for the common case
    of needing just one conformer.

    Args:
        mol: RDKit Mol object
        optimize: Whether to optimize with MMFF force field (default: True)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        Position tensor of shape (num_atoms, 3) with dtype float32,
        where num_atoms == mol.GetNumAtoms().

    Raises:
        FeaturizationError: If conformer generation fails

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")  # Ethanol
        >>> pos = get_positions(mol)
        >>> pos.shape
        torch.Size([3, 3])
    """
    conformers = generate_conformers(
        mol, num_conformers=1, optimize=optimize, random_seed=random_seed
    )
    return conformers[0]


def _optimize_conformers_mmff(mol: Chem.Mol) -> None:
    """Optimize all conformers using MMFF force field.

    Optimization is done in-place on the molecule. If MMFF fails for any reason
    (e.g., missing parameters for unusual elements), the conformers are left
    unoptimized.

    Args:
        mol: RDKit Mol object with conformers already embedded
    """
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    except Exception as e:
        logger.warning("MMFF optimization failed, using unoptimized conformers: %s", e)


def _get_smiles_safe(mol: Chem.Mol) -> str:
    """Safely get SMILES representation of a molecule.

    Args:
        mol: RDKit Mol object

    Returns:
        SMILES string or "unknown" if conversion fails
    """
    try:
        return Chem.MolToSmiles(mol) if mol else "unknown"
    except Exception:
        return "unknown"
