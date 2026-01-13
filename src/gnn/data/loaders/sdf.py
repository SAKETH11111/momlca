"""SDF and MOL file loading utilities."""

import logging
from pathlib import Path

from rdkit import Chem

from ...exceptions import InvalidFileError

logger = logging.getLogger(__name__)


def load_sdf(path: str | Path) -> list[Chem.Mol]:
    """Load molecules from SDF file.

    Args:
        path: Path to SDF file

    Returns:
        List of RDKit Mol objects (may be empty for empty SDF)

    Raises:
        InvalidFileError: If file doesn't exist or cannot be parsed
    """
    path = Path(path)
    if not path.exists():
        raise InvalidFileError(f"File not found: {path}")
    if not path.is_file():
        raise InvalidFileError(f"Not a file: {path}")

    if path.stat().st_size == 0:
        return []

    if path.stat().st_size <= 1024 and not path.read_bytes().strip():
        return []

    supplier = Chem.SDMolSupplier(str(path), removeHs=False)

    molecules = []
    invalid_count = 0
    for idx, mol in enumerate(supplier):
        if mol is None:
            invalid_count += 1
            logger.warning("Skipping invalid molecule %d in %s", idx, path)
            continue
        molecules.append(mol)

    if not molecules:
        raise InvalidFileError(f"Failed to parse any molecules from SDF file: {path}")

    return molecules


def load_mol(path: str | Path) -> Chem.Mol:
    """Load single molecule from MOL file.

    Args:
        path: Path to MOL file

    Returns:
        RDKit Mol object

    Raises:
        InvalidFileError: If file doesn't exist or cannot be parsed
    """
    path = Path(path)
    if not path.exists():
        raise InvalidFileError(f"File not found: {path}")

    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    if mol is None:
        raise InvalidFileError(f"Failed to parse MOL file: {path}")

    return mol
