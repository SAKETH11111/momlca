"""SMILES molecular loading utilities."""

from rdkit import Chem
from rdkit.Chem.SaltRemover import SaltRemover

from ...exceptions import InvalidSMILESError


def load_smiles(smiles: str) -> Chem.Mol:
    """Load and canonicalize a molecule from SMILES string.

    Args:
        smiles: SMILES string representation of molecule

    Returns:
        RDKit Mol object with canonicalized structure and salts removed

    Raises:
        InvalidSMILESError: If SMILES cannot be parsed or is empty
    """
    if not smiles or not smiles.strip():
        raise InvalidSMILESError("Empty SMILES string")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidSMILESError(f"Failed to parse SMILES: {smiles!r}")

    # Remove salts (keep largest fragment)
    remover = SaltRemover()
    mol = remover.StripMol(mol, dontRemoveEverything=True)

    fragments = Chem.GetMolFrags(mol, asMols=True)
    if fragments:
        mol = max(fragments, key=lambda fragment: fragment.GetNumAtoms())

    # Canonicalize by converting to canonical SMILES and back
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    mol = Chem.MolFromSmiles(canonical_smiles)
    if mol is None:
        raise InvalidSMILESError(f"Failed to canonicalize SMILES: {canonical_smiles!r}")

    return mol
