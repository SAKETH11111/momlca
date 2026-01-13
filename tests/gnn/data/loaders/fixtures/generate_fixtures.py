"""Generate test fixture files for SDF/MOL loading tests."""

from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

# Directory where fixtures will be saved
FIXTURES_DIR = Path(__file__).parent

# PFAS molecules for testing
PFAS_SMILES = [
    ("TFA", "C(=O)(C(F)(F)F)O"),
    ("PFBA", "C(=O)(C(C(F)(F)F)(F)F)O"),
    ("PFOA", "C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O"),
]


def generate_single_mol_sdf():
    """Generate SDF file with single molecule."""
    name, smiles = PFAS_SMILES[0]  # TFA
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol, randomSeed=42)  # Generate 3D coords
    mol.SetProp("_Name", name)

    with Chem.SDWriter(str(FIXTURES_DIR / "single_mol.sdf")) as writer:
        writer.write(mol)
    print(f"✓ Created single_mol.sdf ({name})")


def generate_multi_mol_sdf():
    """Generate SDF file with multiple molecules."""
    with Chem.SDWriter(str(FIXTURES_DIR / "multi_mol.sdf")) as writer:
        for name, smiles in PFAS_SMILES:
            mol = Chem.MolFromSmiles(smiles)
            AllChem.EmbedMolecule(mol, randomSeed=42)  # Generate 3D coords
            mol.SetProp("_Name", name)
            writer.write(mol)
    print(f"✓ Created multi_mol.sdf ({len(PFAS_SMILES)} molecules)")


def generate_single_mol_file():
    """Generate MOL file with single molecule."""
    name, smiles = PFAS_SMILES[0]  # TFA
    mol = Chem.MolFromSmiles(smiles)
    AllChem.EmbedMolecule(mol, randomSeed=42)  # Generate 3D coords
    mol.SetProp("_Name", name)

    Chem.MolToMolFile(mol, str(FIXTURES_DIR / "single.mol"))
    print(f"✓ Created single.mol ({name})")


def generate_invalid_sdf():
    """Generate invalid SDF file for error testing."""
    invalid_content = """
INVALID SDF FILE
This is not a valid SDF format
$$$$
"""
    with open(FIXTURES_DIR / "invalid.sdf", "w") as f:
        f.write(invalid_content)
    print("✓ Created invalid.sdf")


def generate_empty_sdf():
    """Generate empty SDF file for empty-file handling tests."""
    (FIXTURES_DIR / "empty.sdf").write_text("")
    print("✓ Created empty.sdf")


if __name__ == "__main__":
    print("Generating test fixtures...")
    generate_single_mol_sdf()
    generate_multi_mol_sdf()
    generate_single_mol_file()
    generate_invalid_sdf()
    generate_empty_sdf()
    print("\nAll fixtures generated successfully!")
