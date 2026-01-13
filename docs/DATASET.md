# PFASBench Dataset

PFASBench is a benchmark dataset for training and evaluating molecular property prediction models on PFAS (per- and polyfluoroalkyl substances).

## Data Sources

The dataset contains curated PFAS molecules with experimentally measured physicochemical properties. Data sources include:

- **Placeholder**: Current sample data is for development/testing purposes
- Future versions will include literature-curated values from peer-reviewed sources

## Properties

| Property | Description | Unit | Range (typical) |
|----------|-------------|------|-----------------|
| logS | Aqueous solubility | log mol/L | -5 to 0 |
| logP | Octanol-water partition coefficient | dimensionless | 0 to 6 |
| pKa | Acid dissociation constant | dimensionless | -3 to 2 |

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Number of molecules | 14 (sample) |
| Property completeness | logS/logP: 14/14, pKa: 6/14 (sample) |
| Carbon chain lengths | C2 to C10 |
| Functional groups | Carboxylic acids, sulfonates, alcohols, amides |

## Curation Steps

1. **SMILES validation**: All SMILES strings parsed with RDKit
2. **Salt removal**: Counter-ions removed, largest fragment retained
3. **Canonicalization**: SMILES converted to canonical form
4. **Featurization**: Atom and bond features extracted using MoleculeFeaturizer
5. **Missing value handling**: NaN values preserved for downstream masking

## Raw Data Format

The raw data is stored in CSV format at `data/pfasbench/raw/pfasbench.csv`:

```csv
smiles,name,logS,logP,pKa
C(=O)(C(F)(F)F)O,TFA,-0.5,0.5,0.5
C(=O)(C(C(F)(F)F)(F)F)O,PFPA,-1.0,1.0,0.6
...
```

### Required columns

- `smiles` (required): SMILES string representation
- `name` (optional): Molecule name/identifier
- `logS` (optional): Aqueous solubility
- `logP` (optional): Partition coefficient
- `pKa` (optional): Acid dissociation constant

## Usage Examples

### Basic usage

```python
from gnn.data import PFASBenchDataset

# Load dataset (first time processes raw data, subsequent loads use cache)
# You can pass either the dataset directory or the base data directory:
dataset = PFASBenchDataset(root="data/")  # uses data/pfasbench/
# dataset = PFASBenchDataset(root="data/pfasbench")

# Check dataset size
print(f"Dataset size: {len(dataset)}")

# Access single molecule
data = dataset[0]
print(f"SMILES: {data.smiles}")
print(f"Atom features shape: {data.x.shape}")
print(f"Labels: {data.y}")
```

### Accessing metadata

```python
# Get SMILES string
smiles = dataset.get_smiles(0)

# Get InChIKey identifier
inchikey = dataset.get_inchikey(0)

# Get property labels as dictionary
labels = dataset.get_labels(0)
print(labels)  # {'logS': -0.5, 'logP': 0.5, 'pKa': 0.5}
```

### Handling missing values

```python
import math

labels = dataset.get_labels(4)  # PFHxA has missing pKa

# Check for missing values
for prop, value in labels.items():
    if math.isnan(value):
        print(f"{prop} is missing")
```

### Using with DataLoader

```python
from torch_geometric.loader import DataLoader

loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    # batch.x: Batched atom features
    # batch.edge_index: Batched edge indices
    # batch.y: Batched labels
    # batch.batch: Atom-to-graph assignment
    pass
```

## Data Object Attributes

Each `Data` object in the dataset contains:

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `x` | float32 | (num_atoms, num_node_features) | Atom features |
| `edge_index` | int64 | (2, num_edges) | Bond connectivity |
| `edge_attr` | float32 | (num_edges, num_edge_features) | Bond features |
| `y` | float32 | (3,) | Property labels [logS, logP, pKa] |
| `smiles` | str | - | Canonical SMILES |
| `name` | str | - | Molecule name |
| `inchikey` | str | - | InChIKey identifier |

## PFAS Families Represented

- **Perfluoroalkyl carboxylic acids (PFCAs)**: TFA, PFPA, PFBA, PFPeA, PFHxA, PFHpA, PFOA, PFNA, PFDA
- **Perfluoroalkyl sulfonic acids (PFSAs)**: PFBS, PFOS
- **Fluorotelomer compounds**: Various structures

## Notes

- Missing property values are stored as NaN and should be masked during loss computation
- InChIKey generation may fail for some structures (empty string returned)
- The dataset uses PyTorch Geometric's InMemoryDataset for efficient caching
- If you provide `data/pfasbench/raw/pfasbench.sdf` (no CSV), the dataset will generate `pfasbench.csv` from SD tags when possible
