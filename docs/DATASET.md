# PFASBench Dataset

PFASBench is a benchmark dataset for training and evaluating molecular property prediction models on PFAS (per- and polyfluoroalkyl substances).

## Data Sources

The dataset is curated from EPA CompTox Dashboard, the authoritative public source for PFAS chemical data.

### Primary Source: EPA CompTox Dashboard

- **URL**: https://comptox.epa.gov/dashboard/
- **PFAS Lists**: https://comptox.epa.gov/dashboard/chemical-lists/PFASSTRUCTv5
- **Size**: 14,735+ PFAS structures (PFASSTRUCTV5 list)
- **Data**: Chemical identifiers, structures (SMILES, InChIKey), and OPERA predictions

### Available PFAS Lists

| List Name | Description | Size |
|-----------|-------------|------|
| PFASSTRUCTV5 | PFAS structures identified by substructure filters | 14,735+ |
| PFASMASTER | Consolidated master list of PFAS | 12,000+ |
| PFASOECD | OECD New Comprehensive Global Database | 3,800+ |

### Property Predictions (OPERA)

OPERA model predictions are one possible source for physicochemical properties:
- **logS**: Water solubility (log mol/L)
- **logP**: Octanol-water partition coefficient
- **pKa**: Acid dissociation constant

OPERA is an open-source QSAR application developed by NIEHS. More info: https://github.com/NIEHS/OPERA

### RDKit-Based Property Computation

When OPERA predictions are not available (e.g., the EPA CompTox CSV export lacks property
columns), the curation script can compute properties using RDKit:

| Property | Method | Reference |
|----------|--------|-----------|
| **logP** | Wildman-Crippen | Wildman & Crippen, J. Chem. Inf. Comput. Sci. 1999, 39(5), 868-873 |
| **logS** | ESOL (Estimated Solubility) | Delaney, J. Chem. Inf. Comput. Sci. 2004, 44(3), 1000-1005 |
| **pKa** | Not computed (left as NaN) | Requires specialized predictor |

To use RDKit-based property computation:
```bash
python scripts/curate_pfasbench.py \
  --input data/pfasbench/raw/comptox_raw/PFASSTRUCTV5.csv \
  --compute-properties
```

The source column will indicate `computed:rdkit` for molecules with computed properties.

### pKa Prediction (pka-predictor-moitessier)

To add/refresh pKa predictions in the curated dataset, run:

```bash
python scripts/add_pka_predictions.py \
  --input data/pfasbench/raw/pfasbench.csv \
  --output data/pfasbench/raw/pfasbench.csv \
  --update-curation-report data/pfasbench/raw/curation_report.json
```

Notes:
- This script fills missing `pKa` values by default and does not overwrite existing values unless `--overwrite-existing` is passed.
- pKa predictions are tracked in the `source` column as `pKa:pka-predictor-moitessier`.

### OPERA Predictions (Optional)

If you have OPERA predictions exported to CSV/TSV, you can merge them into the curated dataset:

```bash
poetry run python scripts/merge_opera_predictions.py \
  --pfasbench data/pfasbench/raw/pfasbench.csv \
  --opera data/pfasbench/raw/opera_predictions.csv \
  --output data/pfasbench/raw/pfasbench_with_opera.csv
```

To prepare OPERA `.smi` input from `pfasbench.csv`:
```bash
poetry run python scripts/prepare_opera_input.py \
  --input data/pfasbench/raw/pfasbench.csv \
  --output data/pfasbench/raw/opera_input.smi
```

## Properties

| Property | Description | Unit | Range (typical) |
|----------|-------------|------|-----------------|
| logS | Aqueous solubility | log mol/L | -10 to 5 |
| logP | Octanol-water partition coefficient | dimensionless | -5 to 10 |
| pKa | Acid dissociation constant | dimensionless | -5 to 20 |

## Dataset Curation

### Obtaining Raw Data

1. **Download from EPA CompTox Dashboard**:
   - Visit https://comptox.epa.gov/dashboard/chemical-lists/PFASSTRUCTv5
   - Click "Export" button
   - Select CSV format
   - Include columns: SMILES, PREFERRED_NAME, INCHIKEY, OPERA predictions
   - Save to `data/pfasbench/raw/comptox_raw/`

2. **Alternative: Use ctx-python API** (requires API key):
   - Request API key by emailing: ccte_api@epa.gov
   - See: https://www.epa.gov/comptox-tools/computational-toxicology-and-exposure-apis-about
   - Library: https://github.com/USEPA/ctx-python

### Running the Curation Script

```bash
# Recommended: Curate from downloaded CompTox export with computed properties
python scripts/curate_pfasbench.py \
  --input data/pfasbench/raw/comptox_raw/PFASSTRUCTV5.csv \
  --compute-properties

# Process multiple files with computed properties
python scripts/curate_pfasbench.py \
  --input-dir data/pfasbench/raw/comptox_raw/ \
  --compute-properties

# With API enrichment (if you have an API key from ccte_api@epa.gov)
export EPA_API_KEY="your_api_key_here"
python scripts/curate_pfasbench.py \
  --input data/pfasbench/raw/comptox_raw/PFASSTRUCTV5.csv \
  --use-api

# Custom output path
python scripts/curate_pfasbench.py --input data.csv --output my_dataset.csv --compute-properties
```

### Curation Pipeline

The curation script performs the following steps:

1. **SMILES Validation**: Parse and validate SMILES strings with RDKit
2. **Salt Removal**: Strip counter-ions using RDKit SaltRemover
3. **Largest Fragment**: Keep only the largest molecular fragment
4. **Canonicalization**: Convert SMILES to canonical form
5. **InChIKey Generation**: Generate unique molecular identifiers
6. **Deduplication**: Remove duplicates by InChIKey (merge properties)
7. **Property Extraction**: Extract logS, logP, pKa from OPERA predictions
8. **Validation**: Verify data quality and generate statistics report

### Output Format

The curated dataset is saved to `data/pfasbench/raw/pfasbench.csv`:

```csv
smiles,name,inchikey,logS,logP,pKa,source
O=C(O)C(F)(F)F,TFA,DTQVDTLACAAQTR-UHFFFAOYSA-N,-0.5,0.5,0.5,PFASSTRUCTV5.csv
O=C(O)C(F)(F)C(F)(F)C(F)(F)F,PFBA,YPJUNDFVDDCYIH-UHFFFAOYSA-N,-1.5,1.5,0.7,PFASSTRUCTV5.csv
...
```

## Dataset Statistics

### Target Statistics

| Metric | Target | Minimum |
|--------|--------|---------|
| Total molecules | 2,000-5,000 | 500 |
| With logS | 50%+ | 30% |
| With logP | 80%+ | 50% |
| With pKa | 30%+ | 10% |
| With all 3 properties | 10%+ | 5% |
| Unique scaffolds | 100+ | 50 |

### Sample Data Statistics (for development)

| Metric | Value |
|--------|-------|
| Number of molecules | 14 |
| With logS | 14 (100%) |
| With logP | 14 (100%) |
| With pKa | 6 (43%) |
| Carbon chain lengths | C2 to C10 |
| Functional groups | Carboxylic acids, sulfonates, alcohols, amides |

> **Note:** The curated dataset from EPA CompTox contains ~14,000+ molecules but requires
> API enrichment (`--use-api`) to obtain property values. The basic CSV export from
> EPA CompTox Dashboard only includes identifiers, not OPERA predictions. See the
> [EPA CCTE API documentation](https://www.epa.gov/comptox-tools/computational-toxicology-and-exposure-apis)
> for API access.

## Raw Data Format

### Required Columns

- `smiles` (required): SMILES string representation
- `name` (optional): Molecule name/identifier
- `inchikey` (optional): InChIKey identifier (generated if missing)
- `logS` (optional): Aqueous solubility
- `logP` (optional): Partition coefficient
- `pKa` (optional): Acid dissociation constant
- `source` (optional): Data source identifier

### Column Name Mappings

The curation script recognizes various column names from CompTox exports:

| Property | Recognized Column Names |
|----------|------------------------|
| SMILES | SMILES, smiles, QSAR_READY_SMILES, CANONICAL_SMILES |
| logS | logS, OPERA_WS_PRED, OPERA_LogS, WATERSOLUBILITY_EXP |
| logP | logP, OPERA_LOGP_PRED, OPERA_LogP, LOGKOW |
| pKa | pKa, OPERA_pKa_acidic, OPERA_pKa_basic, PKA_EXP |

## Usage Examples

### Basic usage

```python
from gnn.data import PFASBenchDataset

# Load dataset (first time processes raw data, subsequent loads use cache)
dataset = PFASBenchDataset(root="data/")

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
    # batch.y: Batched labels (batch_size, 3)
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
| `y` | float32 | (1, 3) | Property labels [logS, logP, pKa] |
| `smiles` | str | - | Canonical SMILES |
| `name` | str | - | Molecule name |
| `inchikey` | str | - | InChIKey identifier |

## PFAS Families Represented

- **Perfluoroalkyl carboxylic acids (PFCAs)**: TFA, PFPA, PFBA, PFPeA, PFHxA, PFHpA, PFOA, PFNA, PFDA
- **Perfluoroalkyl sulfonic acids (PFSAs)**: PFBS, PFOS
- **Fluorotelomer compounds**: Various structures

## Data Quality Requirements

- **No duplicate InChIKeys**: Strictly enforced via deduplication
- **Valid SMILES**: All must parse with RDKit
- **Unit consistency**: logS in log(mol/L), logP unitless, pKa unitless (pH scale)
- **Source tracking**: Know if label is experimental or predicted

To generate a full-dataset QA report:
```bash
poetry run python scripts/generate_pfasbench_data_report.py
```

## Guardrails / Non-goals

- Do **not** generate synthetic/fake property labels
- Do **not** use unlicensed proprietary datasets
- Focus on: real EPA data with documented provenance

## Known Limitations

- **logP and logS are RDKit-computed predictions** (Wildman-Crippen and ESOL respectively)
- **pKa coverage is partial** — predicted using pka-predictor-moitessier (GNN-based)
- Some molecules may fail InChIKey generation
- Very large molecules may have featurization issues

### Property Prediction Methods

| Property | Method | Coverage | Reference |
|----------|--------|----------|-----------|
| **logP** | RDKit Wildman-Crippen | 100% | Wildman & Crippen, JCICS 1999, 39(5), 868-873 |
| **logS** | RDKit ESOL | 100% | Delaney, JCICS 2004, 44(3), 1000-1005 |
| **pKa** | pka-predictor-moitessier | 41.7% | Genzling et al., McGill University |

The current dataset (100% logP, 100% logS, 42% pKa) is suitable for multi-task property prediction.

## References

- [EPA CompTox Dashboard](https://comptox.epa.gov/dashboard/)
- [EPA CompTox API Guide](https://www.epa.gov/comptox-tools/computational-toxicology-and-exposure-apis)
- [PFAS Master List](https://comptox.epa.gov/dashboard/chemical-lists/PFASMASTER)
- [OPERA QSAR Models](https://github.com/NIEHS/OPERA)
- [ctx-python Package](https://github.com/USEPA/ctx-python)

## License / Citation

EPA CompTox data is publicly available for research use. When using this dataset, please cite:

> Williams AJ, et al. The CompTox Chemistry Dashboard: a community data resource for environmental chemistry. J Cheminform. 2017;9:61. doi:10.1186/s13321-017-0247-6

## Notes

- Missing property values are stored as NaN and should be masked during loss computation
- InChIKey generation may fail for some structures (empty string returned)
- The dataset uses PyTorch Geometric's InMemoryDataset for efficient caching
- Sample data (`pfasbench_sample.csv`) is preserved for test fixtures
- If you provide `data/pfasbench/raw/pfasbench.sdf` (no CSV), the dataset will generate `pfasbench.csv` from SD tags
