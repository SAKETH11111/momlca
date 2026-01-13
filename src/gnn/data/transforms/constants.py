"""Feature encoding constants for molecular featurization.

This module defines the allowed values and dimensions for atom and bond features
used in GNN molecular representations.
"""

from rdkit.Chem import rdchem

# Allowed atomic numbers for one-hot encoding
# Covers common organic atoms and PFAS-relevant elements
ALLOWED_ATOMS: list[int] = [
    6,  # C - Carbon
    7,  # N - Nitrogen
    8,  # O - Oxygen
    9,  # F - Fluorine (very common in PFAS)
    15,  # P - Phosphorus
    16,  # S - Sulfur (in sulfonates like PFOS)
    17,  # Cl - Chlorine
    35,  # Br - Bromine
    53,  # I - Iodine
]

# Number of atom types (includes "other" category)
NUM_ATOM_TYPES: int = len(ALLOWED_ATOMS) + 1  # +1 for unknown/other

# Hybridization types for one-hot encoding
HYBRIDIZATION_TYPES: list[rdchem.HybridizationType] = [
    rdchem.HybridizationType.S,
    rdchem.HybridizationType.SP,
    rdchem.HybridizationType.SP2,
    rdchem.HybridizationType.SP3,
    rdchem.HybridizationType.SP3D,
    rdchem.HybridizationType.SP3D2,
]

NUM_HYBRIDIZATION_TYPES: int = len(HYBRIDIZATION_TYPES) + 1  # +1 for unknown

# Bond types for one-hot encoding
BOND_TYPES: list[rdchem.BondType] = [
    rdchem.BondType.SINGLE,
    rdchem.BondType.DOUBLE,
    rdchem.BondType.TRIPLE,
    rdchem.BondType.AROMATIC,
]

NUM_BOND_TYPES: int = len(BOND_TYPES) + 1  # +1 for unknown

# Bond stereo types for one-hot encoding
BOND_STEREO_TYPES: list[rdchem.BondStereo] = [
    rdchem.BondStereo.STEREONONE,
    rdchem.BondStereo.STEREOE,
    rdchem.BondStereo.STEREOZ,
    rdchem.BondStereo.STEREOANY,
]

NUM_STEREO_TYPES: int = len(BOND_STEREO_TYPES) + 1  # +1 for unknown

# Feature dimension documentation
ATOM_FEATURE_DIM: int = (
    NUM_ATOM_TYPES  # Atomic number one-hot (10)
    + 1  # Degree (1)
    + 1  # Formal charge (1)
    + NUM_HYBRIDIZATION_TYPES  # Hybridization one-hot (7)
    + 1  # Is aromatic (1)
    + 1  # Is in ring (1)
    + 1  # Num Hs (1)
)  # Total: 22

BOND_FEATURE_DIM: int = (
    NUM_BOND_TYPES  # Bond type one-hot (5)
    + 1  # Is conjugated (1)
    + 1  # Is in ring (1)
    + NUM_STEREO_TYPES  # Stereo one-hot (5)
)  # Total: 12

# Atom feature layout (indices are relative to returned tensors)
ATOM_ATOMIC_NUMBER_SLICE = slice(0, NUM_ATOM_TYPES)
ATOM_DEGREE_IDX: int = ATOM_ATOMIC_NUMBER_SLICE.stop
ATOM_FORMAL_CHARGE_IDX: int = ATOM_DEGREE_IDX + 1
ATOM_HYBRIDIZATION_SLICE = slice(
    ATOM_FORMAL_CHARGE_IDX + 1,
    ATOM_FORMAL_CHARGE_IDX + 1 + NUM_HYBRIDIZATION_TYPES,
)
ATOM_IS_AROMATIC_IDX: int = ATOM_HYBRIDIZATION_SLICE.stop
ATOM_IS_IN_RING_IDX: int = ATOM_IS_AROMATIC_IDX + 1
ATOM_NUM_HS_IDX: int = ATOM_IS_IN_RING_IDX + 1

# Bond feature layout (indices are relative to returned tensors)
BOND_TYPE_SLICE = slice(0, NUM_BOND_TYPES)
BOND_IS_CONJUGATED_IDX: int = BOND_TYPE_SLICE.stop
BOND_IS_IN_RING_IDX: int = BOND_IS_CONJUGATED_IDX + 1
BOND_STEREO_SLICE = slice(BOND_IS_IN_RING_IDX + 1, BOND_IS_IN_RING_IDX + 1 + NUM_STEREO_TYPES)

# Feature names and dimensions for documentation and debugging
ATOM_FEATURES: list[tuple[str, int]] = [
    ("atomic_number_onehot", NUM_ATOM_TYPES),
    ("degree", 1),
    ("formal_charge", 1),
    ("hybridization_onehot", NUM_HYBRIDIZATION_TYPES),
    ("is_aromatic", 1),
    ("is_in_ring", 1),
    ("num_hs", 1),
]

BOND_FEATURES: list[tuple[str, int]] = [
    ("bond_type_onehot", NUM_BOND_TYPES),
    ("is_conjugated", 1),
    ("is_in_ring", 1),
    ("stereo_onehot", NUM_STEREO_TYPES),
]
