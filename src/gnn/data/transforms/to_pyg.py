"""PyG Data conversion utilities for molecular graphs.

This module provides functions to convert RDKit Mol objects into
PyTorch Geometric (PyG) Data and Batch objects for use with GNN models.
"""

from __future__ import annotations

import contextlib
from typing import Any

import torch
from rdkit import Chem
from rdkit.Chem import AllChem as _AllChem
from torch_geometric.data import Batch, Data

from gnn.exceptions import FeaturizationError

from .featurizer import MoleculeFeaturizer

AllChem: Any = _AllChem


def _normalize_target(y: torch.Tensor | float | int) -> torch.Tensor:
    if isinstance(y, (float, int)):
        return torch.tensor([y], dtype=torch.float32)

    if y.ndim == 0:
        return y.reshape(1)
    if y.ndim == 1:
        if y.numel() == 1:
            return y
        return y.unsqueeze(0)
    if y.shape[0] != 1:
        raise ValueError("Tensor target for a single molecule must have leading dimension 1")
    return y


def _positions_from_conformer(mol: Chem.Mol) -> torch.Tensor | None:
    if mol.GetNumConformers() == 0:
        return None

    conformer = mol.GetConformer()
    positions = conformer.GetPositions()
    pos = torch.tensor(positions, dtype=torch.float32)
    if pos.shape != (mol.GetNumAtoms(), 3):
        raise FeaturizationError(
            f"Conformer positions have unexpected shape {tuple(pos.shape)}; "
            f"expected ({mol.GetNumAtoms()}, 3)"
        )
    return pos


def _generate_positions(
    mol: Chem.Mol,
    *,
    optimize: bool,
    random_seed: int | None,
) -> torch.Tensor:
    num_input_atoms = mol.GetNumAtoms()

    mol_with_h = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    if random_seed is not None:
        params.randomSeed = int(random_seed)

    conf_id = AllChem.EmbedMolecule(mol_with_h, params)
    if conf_id < 0:
        raise FeaturizationError("Failed to embed 3D conformer with ETKDGv3")

    if optimize:
        with contextlib.suppress(Exception):
            AllChem.MMFFOptimizeMolecule(mol_with_h, confId=conf_id)

    conformer = mol_with_h.GetConformer(conf_id)
    positions = conformer.GetPositions()
    return torch.tensor(positions[:num_input_atoms], dtype=torch.float32)


def mol_to_pyg_data(
    mol: Chem.Mol,
    include_pos: bool = False,
    generate_pos_if_missing: bool = True,
    pos_optimize: bool = True,
    pos_random_seed: int | None = None,
    y: torch.Tensor | float | int | None = None,
) -> Data:
    """Convert RDKit Mol to PyTorch Geometric Data object.

    This function extracts atom features, bond connectivity, and bond features
    from an RDKit molecule and creates a PyG Data object suitable for GNN input.

    Args:
        mol: RDKit Mol object
        include_pos: Whether to include 3D positions (default: False). When True,
            this function uses existing conformer coordinates if present; otherwise
            it can optionally generate a conformer (see generate_pos_if_missing).
        generate_pos_if_missing: When include_pos=True and the molecule has no
            conformers, generate a conformer using ETKDGv3 (default: True).
        pos_optimize: When generating positions, attempt MMFF optimization
            (default: True). Optimization failure does not raise.
        pos_random_seed: Optional random seed passed to RDKit ETKDGv3 parameters
            for reproducible position generation. If None, RDKit defaults apply.
        y: Optional target value(s) for supervised learning. Accepts:
            - float/int: Converted to float32 tensor of shape (1,)
            - torch.Tensor: Normalized to shape (1, ...) for graph-level labels

    Returns:
        PyG Data object with attributes:
            - x: Atom features tensor (num_atoms, 22)
            - edge_index: Bond connectivity (2, num_edges) - bidirectional
            - edge_attr: Bond features tensor (num_edges, 12)
            - num_nodes: Number of atoms
            - smiles: Canonical SMILES string (for traceability)
            - pos: 3D positions (num_atoms, 3) if include_pos=True
            - y: Target value(s) if provided

    Raises:
        FeaturizationError: If featurization or conformer generation fails

    Example:
        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles("CCO")  # Ethanol
        >>> data = mol_to_pyg_data(mol)
        >>> data.x.shape
        torch.Size([3, 22])
        >>> data.edge_index.shape
        torch.Size([2, 4])  # 2 bonds * 2 directions
    """
    if mol is None:
        raise FeaturizationError("Cannot convert to PyG Data: molecule is None")

    # Get features from MoleculeFeaturizer
    featurizer = MoleculeFeaturizer()
    try:
        features = featurizer.featurize(mol)
    except Exception as e:
        raise FeaturizationError(f"Featurization failed: {e}") from e

    # Create Data object with core attributes
    data = Data(
        x=features["x"],
        edge_index=features["edge_index"],
        edge_attr=features["edge_attr"],
        num_nodes=features["num_nodes"],
    )

    # Add 3D positions if requested
    if include_pos:
        pos = _positions_from_conformer(mol)
        if pos is None and generate_pos_if_missing:
            pos = _generate_positions(mol, optimize=pos_optimize, random_seed=pos_random_seed)
        if pos is None:
            raise FeaturizationError(
                "include_pos=True but molecule has no conformers; "
                "set generate_pos_if_missing=True or provide 3D coordinates"
            )
        data.pos = pos

    # Add canonical SMILES for traceability
    try:
        data.smiles = Chem.MolToSmiles(mol, canonical=True)
    except Exception as e:
        raise FeaturizationError(f"Failed to canonicalize SMILES: {e}") from e

    # Add target value if provided
    if y is not None:
        data.y = _normalize_target(y)

    return data


def mols_to_pyg_batch(
    mols: list[Chem.Mol],
    include_pos: bool = False,
    generate_pos_if_missing: bool = True,
    pos_optimize: bool = True,
    pos_random_seed: int | None = None,
    ys: list[float] | list[int] | torch.Tensor | None = None,
) -> Batch:
    """Convert list of RDKit Mols to PyG Batch.

    This function converts multiple molecules to individual Data objects
    and combines them into a single Batch using PyG's Batch.from_data_list().
    The resulting Batch provides efficient batched operations for GNNs.

    Args:
        mols: List of RDKit Mol objects (must be non-empty)
        include_pos: Whether to generate and include 3D positions for all
            molecules (default: False)
        generate_pos_if_missing: When include_pos=True, generate positions for
            molecules that do not already have conformers (default: True).
        pos_optimize: Whether to attempt MMFF optimization when generating
            positions (default: True).
        pos_random_seed: Optional random seed passed to RDKit ETKDGv3 parameters.
        ys: Optional target values (one per molecule). Accepts:
            - List of float/int: One target per molecule
            - torch.Tensor: Shape (num_molecules,) or (num_molecules, num_targets)

    Returns:
        PyG Batch object with batched attributes:
            - x: Concatenated atom features (total_atoms, 22)
            - edge_index: Concatenated + shifted edge indices (2, total_edges)
            - edge_attr: Concatenated bond features (total_edges, 12)
            - batch: Atom-to-graph assignment (total_atoms,)
            - ptr: Graph boundaries (num_graphs + 1,)
            - pos: Concatenated positions (total_atoms, 3) if include_pos=True
            - y: Batched targets if provided
            - smiles: List of canonical SMILES strings

    Raises:
        ValueError: If mols is empty
        FeaturizationError: If any molecule fails to featurize

    Example:
        >>> from rdkit import Chem
        >>> mols = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("c1ccccc1")]
        >>> batch = mols_to_pyg_batch(mols, ys=[1.0, 2.0])
        >>> batch.num_graphs
        2
        >>> batch.batch  # Atom-to-graph assignment
        tensor([0, 0, 0, 1, 1, 1, 1, 1, 1])
    """
    if not mols:
        raise ValueError("Cannot create batch from empty molecule list")

    if ys is not None:
        if isinstance(ys, torch.Tensor):
            if ys.ndim == 0:
                raise ValueError("ys tensor must have a leading dimension (num_molecules, ...)")
            if ys.shape[0] != len(mols):
                raise ValueError("ys tensor length must match mols length")
        else:
            if len(ys) != len(mols):
                raise ValueError("ys length must match mols length")

    data_list = []
    for i, mol in enumerate(mols):
        # Get target for this molecule if provided
        y = ys[i] if ys is not None else None

        data = mol_to_pyg_data(
            mol,
            include_pos=include_pos,
            generate_pos_if_missing=generate_pos_if_missing,
            pos_optimize=pos_optimize,
            pos_random_seed=pos_random_seed,
            y=y,
        )
        data_list.append(data)

    return Batch.from_data_list(data_list)
