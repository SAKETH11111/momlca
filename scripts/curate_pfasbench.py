"""Curate PFASBench dataset from EPA CompTox Dashboard.

This script curates PFAS molecular data from EPA CompTox Dashboard exports into
a clean, deduplicated dataset for model training and benchmarking.

Data Sources:
    - EPA CompTox Dashboard PFASSTRUCTV5 list (14,735+ structures)
    - OPERA predictions for physicochemical properties
    - Download from: https://comptox.epa.gov/dashboard/chemical-lists/PFASSTRUCTv5

Usage:
    # Process a pre-downloaded CompTox CSV export
    python scripts/curate_pfasbench.py --input data/pfasbench/raw/comptox_raw/PFASSTRUCTV5.csv

    # Use batch download mode (processes multiple files)
    python scripts/curate_pfasbench.py --input-dir data/pfasbench/raw/comptox_raw/

    # With API enrichment (requires API key)
    python scripts/curate_pfasbench.py --input data/comptox_export.csv --use-api --api-key YOUR_KEY

References:
    - EPA CompTox Dashboard: https://comptox.epa.gov/dashboard/
    - PFAS Lists: https://comptox.epa.gov/dashboard/chemical-lists/PFASSTRUCTv5
    - ctx-python: https://github.com/USEPA/ctx-python
    - OPERA: https://github.com/NIEHS/OPERA
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors, SaltRemover
from rdkit.Chem.inchi import MolToInchiKey
from rdkit.Chem.MolStandardize import rdMolStandardize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# RDKit-based Property Computation
# =============================================================================


def compute_logp(mol: Chem.Mol) -> float:
    """Compute logP using Wildman-Crippen method.

    The Wildman-Crippen method is a fragment-based approach that estimates
    the octanol-water partition coefficient (logP) from atom contributions.

    Reference:
        Wildman, S. A., & Crippen, G. M. (1999). Prediction of Physicochemical
        Parameters by Atomic Contributions. J. Chem. Inf. Comput. Sci., 39(5), 868-873.

    Args:
        mol: RDKit Mol object.

    Returns:
        Predicted logP value, or NaN on failure.
    """
    try:
        return Descriptors.MolLogP(mol)
    except Exception:
        return float("nan")


def compute_esol_logs(mol: Chem.Mol) -> float:
    """Compute aqueous solubility (logS) using ESOL model.

    The ESOL (Estimated SOLubility) model from Delaney (2004) predicts
    aqueous solubility from molecular structure using a simple linear
    equation based on four molecular descriptors.

    Reference:
        Delaney, J. S. (2004). ESOL: Estimating Aqueous Solubility
        Directly from Molecular Structure. J. Chem. Inf. Comput. Sci.,
        44(3), 1000-1005. https://doi.org/10.1021/ci034243x

    Args:
        mol: RDKit Mol object.

    Returns:
        Predicted logS in log(mol/L), or NaN on failure.
    """
    try:
        # Molecular weight
        mw = Descriptors.MolWt(mol)

        # Wildman-Crippen logP
        logp = Descriptors.MolLogP(mol)

        # Number of rotatable bonds
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)

        # Aromatic proportion (aromatic atoms / heavy atoms)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        heavy_atoms = mol.GetNumHeavyAtoms()
        aromatic_proportion = aromatic_atoms / heavy_atoms if heavy_atoms > 0 else 0

        # ESOL equation coefficients from Delaney 2004
        # logS = 0.16 - 0.63*cLogP - 0.0062*MW + 0.066*RB - 0.74*AP
        logs = (
            0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rotatable_bonds - 0.74 * aromatic_proportion
        )

        return logs

    except Exception:
        return float("nan")


def compute_properties_for_smiles(smiles: str) -> dict[str, float]:
    """Compute RDKit-based properties for a SMILES string.

    Args:
        smiles: Canonical SMILES string.

    Returns:
        Dictionary with computed logP and logS values.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"logP": float("nan"), "logS": float("nan")}

        return {
            "logP": compute_logp(mol),
            "logS": compute_esol_logs(mol),
        }
    except Exception:
        return {"logP": float("nan"), "logS": float("nan")}


# Constants
DEFAULT_OUTPUT_PATH = Path("data/pfasbench/raw/pfasbench.csv")
DEFAULT_SAMPLE_PATH = Path("data/pfasbench/raw/pfasbench_sample.csv")
RAW_DOWNLOAD_DIR = Path("data/pfasbench/raw/comptox_raw")
REPORT_PATH = Path("data/pfasbench/raw/curation_report.json")

# Property column name mappings from CompTox exports
PROPERTY_MAPPINGS: dict[str, list[str]] = {
    "logS": [
        "logS",  # Direct column name
        "OPERA_WS_PRED",  # OPERA water solubility prediction
        "OPERA_LogS",
        "LogS_OPERA",
        "water_solubility_pred",
        "WS_OPERA",
        "WATERSOLUBILITY_EXP",
        "LOG_S",
        "logs",
        "LogS",
    ],
    "logP": [
        "logP",  # Direct column name
        "OPERA_LOGP_PRED",  # OPERA logP prediction
        "OPERA_LogP",
        "LogP_OPERA",
        "logp_pred",
        "LOGKOW",  # Experimental logP
        "LOG_KOW_EXP",
        "LOGP_EXP",
        "LOG_P",
        "logp",
        "LogP",
        "logKow",
        "ALOGP",
    ],
    "pKa": [
        "pKa",  # Direct column name
        "OPERA_pKa_acidic",  # OPERA pKa prediction
        "OPERA_pKa_basic",
        "OPERA_PKA_PRED",
        "pKa_OPERA",
        "pKa_pred",
        "PKA_EXP",
        "pka",
        "PKA",
    ],
}

# SMILES column mappings from CompTox exports
SMILES_COLUMNS = [
    "SMILES",
    "smiles",
    "QSAR_READY_SMILES",
    "CANONICAL_SMILES",
    "MS_READY_SMILES",
    "INPUT_SMILES",
    "Smiles",
]

# Name/ID column mappings
NAME_COLUMNS = [
    "PREFERRED_NAME",
    "PREFERRED NAME",  # CompTox exports may use space
    "preferred_name",
    "CASRN",
    "casrn",
    "NAME",
    "name",
    "CHEMICAL_NAME",
    "chemical_name",
    "DTXSID",
    "dtxsid",
]

# InChIKey column mappings
INCHIKEY_COLUMNS = [
    "INCHIKEY",
    "InChIKey",
    "inchikey",
    "INCHI_KEY",
    "INCHI KEY",  # CompTox exports may use space
]


@dataclass
class CurationStats:
    """Container for curation statistics."""

    input_count: int = 0
    valid_smiles: int = 0
    invalid_smiles: int = 0
    salt_stripped: int = 0
    duplicates_removed: int = 0
    final_count: int = 0
    logS_available: int = 0
    logP_available: int = 0
    pKa_available: int = 0
    all_props_available: int = 0
    sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    processing_time_sec: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class MoleculeProcessor:
    """Process and clean molecular structures."""

    def __init__(self) -> None:
        """Initialize the molecule processor."""
        self.salt_remover = SaltRemover.SaltRemover()
        self.uncharger = rdMolStandardize.Uncharger()
        self.stats = CurationStats()

    def parse_smiles(self, smiles: str) -> Chem.Mol | None:
        """Parse and validate a SMILES string.

        Args:
            smiles: Input SMILES string.

        Returns:
            RDKit Mol object or None if invalid.
        """
        if pd.isna(smiles) or not str(smiles).strip():
            return None

        smiles_clean = str(smiles).strip()
        try:
            mol = Chem.MolFromSmiles(smiles_clean)
            if mol is None:
                return None
            return mol
        except Exception:
            return None

    def strip_salts(self, mol: Chem.Mol) -> Chem.Mol:
        """Remove salts and keep the largest fragment.

        Args:
            mol: RDKit Mol object.

        Returns:
            Processed Mol object with salts removed.
        """
        try:
            # First try RDKit's salt remover
            mol_stripped = self.salt_remover.StripMol(mol, dontRemoveEverything=True)
        except Exception:
            mol_stripped = mol

        # Then keep largest fragment
        try:
            frags = Chem.GetMolFrags(mol_stripped, asMols=True, sanitizeFrags=True)
            if len(frags) > 1:
                # Sort by heavy atom count, keep largest
                frags_sorted = sorted(frags, key=lambda x: x.GetNumHeavyAtoms(), reverse=True)
                self.stats.salt_stripped += 1
                return frags_sorted[0]
            return mol_stripped
        except Exception:
            return mol_stripped

    def canonicalize(self, mol: Chem.Mol) -> str:
        """Generate canonical SMILES.

        Args:
            mol: RDKit Mol object.

        Returns:
            Canonical SMILES string.
        """
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return ""

    def generate_inchikey(self, mol: Chem.Mol) -> str:
        """Generate InChIKey for a molecule.

        Args:
            mol: RDKit Mol object.

        Returns:
            InChIKey string or empty string on failure.
        """
        try:
            return MolToInchiKey(mol)
        except Exception:
            return ""

    def process_molecule(self, smiles: str, name: str = "") -> dict[str, Any] | None:
        """Process a single molecule through the cleaning pipeline.

        Args:
            smiles: Input SMILES string.
            name: Molecule name/identifier.

        Returns:
            Dictionary with processed molecule data, or None if invalid.
        """
        mol = self.parse_smiles(smiles)
        if mol is None:
            self.stats.invalid_smiles += 1
            return None

        self.stats.valid_smiles += 1

        # Strip salts and keep largest fragment
        mol_clean = self.strip_salts(mol)

        # Generate canonical SMILES
        canonical_smiles = self.canonicalize(mol_clean)
        if not canonical_smiles:
            return None

        # Always generate InChIKey from the CLEANED molecule
        # This ensures proper deduplication after salt stripping
        # (original InChIKeys may differ for salts that become identical after stripping)
        inchikey = self.generate_inchikey(mol_clean)

        if not inchikey:
            return None

        return {
            "smiles": canonical_smiles,
            "name": str(name) if name and not pd.isna(name) else "",
            "inchikey": inchikey,
            "mol": mol_clean,
        }


class CompToxAPIClient:
    """Client for EPA CCTE API to fetch chemical properties."""

    BASE_URL = "https://api.epa.gov/ccte/api/v1"

    def __init__(self, api_key: str, batch_size: int = 100) -> None:
        """Initialize the API client.

        Args:
            api_key: EPA CCTE API key.
            batch_size: Number of DTXSIDs per request.
        """
        self.api_key = api_key
        self.batch_size = batch_size
        self.headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def fetch_physchem_properties(self, dtxsids: list[str]) -> dict[str, dict[str, float]]:
        """Fetch physicochemical properties for a list of DTXSIDs.

        Args:
            dtxsids: List of DTXSID strings.

        Returns:
            Dictionary mapping DTXSID to property dictionary.
        """
        results: dict[str, dict[str, float]] = {}

        # Split into batches
        for i in range(0, len(dtxsids), self.batch_size):
            batch = dtxsids[i : i + self.batch_size]
            logger.info("Fetching properties for batch of %d chemicals...", len(batch))

            try:
                # POST /physchem-predictions/search/by-dtxsid
                url = f"{self.BASE_URL}/physchem-predictions/search/by-dtxsid"
                response = requests.post(url, headers=self.headers, json=batch, timeout=30)

                if response.status_code == 200:
                    data = response.json()
                    # Response is a list of objects containing DTXSID and predictions
                    for item in data:
                        dtxsid = item.get("dtxsid")
                        if not dtxsid:
                            continue

                        props = {}
                        predictions = item.get("predictions", [])
                        for pred in predictions:
                            # Map OPERA property names to our internal names
                            prop_name = pred.get("propertyName")
                            value = pred.get("value")

                            if prop_name == "Water solubility" and value is not None:
                                props["logS"] = float(value)
                            elif prop_name == "LogP" and value is not None:
                                props["logP"] = float(value)
                            elif prop_name == "pKa" and value is not None:
                                props["pKa"] = float(value)

                        if props:
                            results[dtxsid] = props
                else:
                    logger.error(
                        "API request failed with status %d: %s", response.status_code, response.text
                    )

                # Add small delay to respect rate limits if needed
                time.sleep(0.1)

            except Exception as e:
                logger.error("Error fetching properties from API: %s", e)

        return results


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Find the first matching column name from candidates.

    Args:
        df: DataFrame to search.
        candidates: List of potential column names.

    Returns:
        First matching column name, or None if no match.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def extract_property_value(row: pd.Series, prop_name: str) -> float:
    """Extract property value from a row using known column mappings.

    Args:
        row: DataFrame row.
        prop_name: Target property name (logS, logP, pKa).

    Returns:
        Property value as float, or NaN if not found.
    """
    candidates = PROPERTY_MAPPINGS.get(prop_name, [])
    for col in candidates:
        if col in row.index:
            val = row[col]
            if not pd.isna(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
    return float("nan")


def load_comptox_export(filepath: Path) -> pd.DataFrame:
    """Load a CompTox Dashboard export file.

    Supports CSV, TSV, and Excel formats.

    Args:
        filepath: Path to the export file.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If file format is not supported.
    """
    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(filepath, low_memory=False)
    elif suffix == ".tsv":
        return pd.read_csv(filepath, sep="\t", low_memory=False)
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def process_comptox_dataframe(
    df: pd.DataFrame,
    processor: MoleculeProcessor,
    source_name: str = "",
) -> list[dict[str, Any]]:
    """Process a CompTox export DataFrame into curated records.

    Args:
        df: CompTox DataFrame.
        processor: MoleculeProcessor instance.
        source_name: Source identifier for tracking.

    Returns:
        List of processed molecule records.
    """
    records: list[dict[str, Any]] = []
    processor.stats.input_count += len(df)

    # Find SMILES column
    smiles_col = find_column(df, SMILES_COLUMNS)
    if smiles_col is None:
        logger.error("No SMILES column found in %s", source_name)
        processor.stats.errors.append(f"No SMILES column in {source_name}")
        return records

    # Find name column
    name_col = find_column(df, NAME_COLUMNS)

    # Find InChIKey column
    inchikey_col = find_column(df, INCHIKEY_COLUMNS)

    # Find DTXSID column specifically for API use
    dtxsid_col = find_column(df, ["DTXSID", "dtxsid"])

    logger.info(
        "Processing %d rows from %s (SMILES=%s, Name=%s, InChIKey=%s, DTXSID=%s)",
        len(df),
        source_name,
        smiles_col,
        name_col,
        inchikey_col,
        dtxsid_col,
    )

    for _idx, row in df.iterrows():
        smiles = row[smiles_col]
        name = row[name_col] if name_col else ""
        dtxsid = str(row[dtxsid_col]).strip() if dtxsid_col and not pd.isna(row[dtxsid_col]) else ""

        result = processor.process_molecule(smiles, name)
        if result is None:
            continue

        # Store DTXSID for API enrichment
        result["dtxsid"] = dtxsid

        # Extract property values
        result["logS"] = extract_property_value(row, "logS")
        result["logP"] = extract_property_value(row, "logP")
        result["pKa"] = extract_property_value(row, "pKa")
        result["source"] = source_name

        # Remove the mol object before storing
        del result["mol"]
        records.append(result)

    return records


def deduplicate_by_inchikey(
    records: list[dict[str, Any]],
    stats: CurationStats,
) -> list[dict[str, Any]]:
    """Deduplicate records by InChIKey.

    For duplicates, properties are merged, preferring non-NaN over NaN values.

    Args:
        records: List of molecule records.
        stats: CurationStats to update.

    Returns:
        Deduplicated list of records.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        inchikey = record["inchikey"]
        if inchikey not in grouped:
            grouped[inchikey] = []
        grouped[inchikey].append(record)

    deduplicated: list[dict[str, Any]] = []
    duplicates_merged = 0

    for _inchikey, group in grouped.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Merge properties from duplicates
            merged = group[0].copy()
            duplicates_merged += len(group) - 1

            for prop in ["logS", "logP", "pKa"]:
                # Collect all non-NaN values
                values = [r[prop] for r in group if not np.isnan(r[prop])]
                if values:
                    # Use mean of available values
                    merged[prop] = float(np.mean(values))
                else:
                    merged[prop] = float("nan")

            # Combine sources
            sources = list({r.get("source", "") for r in group if r.get("source")})
            merged["source"] = ";".join(sources) if sources else ""

            deduplicated.append(merged)

    stats.duplicates_removed = duplicates_merged
    logger.info(
        "Deduplication: %d unique molecules, %d duplicates merged",
        len(deduplicated),
        duplicates_merged,
    )

    return deduplicated


def compute_statistics(records: list[dict[str, Any]], stats: CurationStats) -> None:
    """Compute final statistics for the curated dataset.

    Args:
        records: Final list of records.
        stats: CurationStats to update.
    """
    stats.final_count = len(records)

    for record in records:
        has_logS = not np.isnan(record.get("logS", float("nan")))
        has_logP = not np.isnan(record.get("logP", float("nan")))
        has_pKa = not np.isnan(record.get("pKa", float("nan")))

        if has_logS:
            stats.logS_available += 1
        if has_logP:
            stats.logP_available += 1
        if has_pKa:
            stats.pKa_available += 1
        if has_logS and has_logP and has_pKa:
            stats.all_props_available += 1


def save_dataset(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save curated records to CSV.

    Args:
        records: List of molecule records.
        output_path: Path for output CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create DataFrame with consistent column order
    columns = ["smiles", "name", "inchikey", "logS", "logP", "pKa", "source"]
    df = pd.DataFrame(records, columns=columns)

    # Sort by name/smiles for reproducibility
    df = df.sort_values(by=["name", "smiles"]).reset_index(drop=True)

    df.to_csv(output_path, index=False)
    logger.info("Saved %d molecules to %s", len(df), output_path)


def save_report(stats: CurationStats, report_path: Path) -> None:
    """Save curation statistics report.

    Args:
        stats: CurationStats object.
        report_path: Path for JSON report.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": stats.timestamp,
        "processing_time_sec": stats.processing_time_sec,
        "input": {
            "total_rows": stats.input_count,
            "sources": stats.sources,
        },
        "processing": {
            "valid_smiles": stats.valid_smiles,
            "invalid_smiles": stats.invalid_smiles,
            "salt_stripped": stats.salt_stripped,
            "duplicates_removed": stats.duplicates_removed,
        },
        "output": {
            "final_count": stats.final_count,
            "properties": {
                "logS_available": stats.logS_available,
                "logP_available": stats.logP_available,
                "pKa_available": stats.pKa_available,
                "all_properties": stats.all_props_available,
            },
            "property_coverage": {
                "logS_pct": (
                    100.0 * stats.logS_available / stats.final_count
                    if stats.final_count > 0
                    else 0.0
                ),
                "logP_pct": (
                    100.0 * stats.logP_available / stats.final_count
                    if stats.final_count > 0
                    else 0.0
                ),
                "pKa_pct": (
                    100.0 * stats.pKa_available / stats.final_count
                    if stats.final_count > 0
                    else 0.0
                ),
            },
        },
        "errors": stats.errors,
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Saved curation report to %s", report_path)


def backup_existing_sample(output_path: Path, sample_path: Path) -> None:
    """Backup existing sample data before overwriting.

    Args:
        output_path: Path to main dataset.
        sample_path: Path to backup as sample.
    """
    if output_path.exists():
        # Check if it's the small sample dataset
        try:
            df = pd.read_csv(output_path)
            if len(df) <= 20:  # Likely the original sample
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(sample_path, index=False)
                logger.info(
                    "Backed up existing %d-molecule sample to %s",
                    len(df),
                    sample_path,
                )
        except Exception as e:
            logger.warning("Could not backup existing file: %s", e)


def main() -> int:
    """Main entry point for the curation script.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    parser = argparse.ArgumentParser(
        description="Curate PFASBench dataset from EPA CompTox Dashboard exports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a single CompTox CSV export
    python scripts/curate_pfasbench.py --input comptox_export.csv

    # Process all files in a directory
    python scripts/curate_pfasbench.py --input-dir data/pfasbench/raw/comptox_raw/

    # Custom output path
    python scripts/curate_pfasbench.py --input data.csv --output my_dataset.csv
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        help="Path to CompTox export file (CSV, TSV, or Excel)",
    )
    parser.add_argument(
        "--input-dir",
        "-d",
        type=Path,
        help="Directory containing CompTox export files to process",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=REPORT_PATH,
        help=f"Curation report JSON path (default: {REPORT_PATH})",
    )
    parser.add_argument(
        "--min-molecules",
        type=int,
        default=500,
        help="Minimum number of molecules required (default: 500)",
    )
    parser.add_argument(
        "--keep-sample",
        action="store_true",
        help="Backup existing sample dataset before overwriting",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use EPA CCTE API to fetch missing properties",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("EPA_API_KEY"),
        help="EPA CCTE API key (can also set EPA_API_KEY env var)",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=100,
        help="Number of chemicals per API request (default: 100)",
    )
    parser.add_argument(
        "--compute-properties",
        "-c",
        action="store_true",
        help="Compute logP and logS using RDKit when values are missing. "
        "Uses Wildman-Crippen for logP and ESOL (Delaney 2004) for logS.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input arguments
    if not args.input and not args.input_dir:
        parser.error("Either --input or --input-dir is required")

    start_time = time.time()
    processor = MoleculeProcessor()
    all_records: list[dict[str, Any]] = []

    # Backup existing sample if requested
    if args.keep_sample:
        backup_existing_sample(args.output, DEFAULT_SAMPLE_PATH)

    # Process input file(s)
    input_files: list[Path] = []

    if args.input:
        if not args.input.exists():
            logger.error("Input file not found: %s", args.input)
            return 1
        input_files.append(args.input)

    if args.input_dir:
        if not args.input_dir.exists():
            logger.error("Input directory not found: %s", args.input_dir)
            return 1
        # Find all supported files
        for ext in ("*.csv", "*.tsv", "*.xlsx", "*.xls"):
            input_files.extend(args.input_dir.glob(ext))

    if not input_files:
        logger.error("No input files found to process")
        return 1

    logger.info("Processing %d input file(s)", len(input_files))

    for filepath in input_files:
        try:
            logger.info("Loading %s...", filepath)
            df = load_comptox_export(filepath)
            records = process_comptox_dataframe(df, processor, source_name=filepath.name)
            all_records.extend(records)
            processor.stats.sources.append(filepath.name)
        except Exception as e:
            logger.error("Error processing %s: %s", filepath, e)
            processor.stats.errors.append(f"Error in {filepath.name}: {str(e)}")

    if not all_records:
        logger.error("No valid molecules extracted from input files")
        return 1

    # Deduplicate
    logger.info("Deduplicating %d records by InChIKey...", len(all_records))
    deduplicated = deduplicate_by_inchikey(all_records, processor.stats)

    # API Enrichment
    if args.use_api:
        if not args.api_key:
            logger.error(
                "API key required for --use-api. Use --api-key or set EPA_API_KEY env var."
            )
            return 1

        # Identify unique DTXSIDs with missing properties
        dtxsids_to_fetch = set()
        for record in deduplicated:
            dtxsid = record.get("dtxsid")
            if dtxsid and (
                np.isnan(record.get("logS", np.nan))
                or np.isnan(record.get("logP", np.nan))
                or np.isnan(record.get("pKa", np.nan))
            ):
                dtxsids_to_fetch.add(dtxsid)

        if dtxsids_to_fetch:
            logger.info("Enriching %d molecules via EPA CCTE API...", len(dtxsids_to_fetch))
            client = CompToxAPIClient(api_key=args.api_key, batch_size=args.api_batch_size)
            api_results = client.fetch_physchem_properties(list(dtxsids_to_fetch))

            # Update records
            enriched_count = 0
            for record in deduplicated:
                dtxsid = record.get("dtxsid")
                if dtxsid in api_results:
                    props = api_results[dtxsid]
                    for prop, val in props.items():
                        # Only fill if currently NaN
                        if np.isnan(record.get(prop, np.nan)):
                            record[prop] = val
                            enriched_count += 1
            logger.info("Updated %d property values from API", enriched_count)
        else:
            logger.info("No missing properties to fetch from API")

    # RDKit Property Computation
    if args.compute_properties:
        logger.info("Computing properties with RDKit for molecules with missing values...")
        computed_logp_count = 0
        computed_logs_count = 0

        for record in deduplicated:
            smiles = record.get("smiles", "")
            needs_logp = np.isnan(record.get("logP", np.nan))
            needs_logs = np.isnan(record.get("logS", np.nan))

            if needs_logp or needs_logs:
                computed = compute_properties_for_smiles(smiles)

                if needs_logp and not np.isnan(computed["logP"]):
                    record["logP"] = computed["logP"]
                    computed_logp_count += 1

                if needs_logs and not np.isnan(computed["logS"]):
                    record["logS"] = computed["logS"]
                    computed_logs_count += 1

                # Update source to indicate computed values
                source = record.get("source", "")
                if needs_logp or needs_logs:
                    if source and "computed:rdkit" not in source:
                        record["source"] = f"{source};computed:rdkit"
                    elif not source:
                        record["source"] = "computed:rdkit"

        logger.info(
            "Computed %d logP and %d logS values using RDKit",
            computed_logp_count,
            computed_logs_count,
        )

    # Compute final statistics
    compute_statistics(deduplicated, processor.stats)

    # Check minimum count
    if len(deduplicated) < args.min_molecules:
        logger.warning(
            "Only %d molecules extracted, below minimum threshold of %d",
            len(deduplicated),
            args.min_molecules,
        )

    # Save results
    save_dataset(deduplicated, args.output)

    processor.stats.processing_time_sec = time.time() - start_time
    save_report(processor.stats, args.report)

    # Print summary
    print("\n" + "=" * 60)
    print("PFASBench Curation Summary")
    print("=" * 60)
    print(f"Input files:          {len(input_files)}")
    print(f"Total rows processed: {processor.stats.input_count}")
    print(f"Valid SMILES:         {processor.stats.valid_smiles}")
    print(f"Invalid SMILES:       {processor.stats.invalid_smiles}")
    print(f"Salts stripped:       {processor.stats.salt_stripped}")
    print(f"Duplicates merged:    {processor.stats.duplicates_removed}")
    print(f"Final molecules:      {processor.stats.final_count}")
    print("-" * 60)
    print("Property Coverage:")
    print(
        f"  logS: {processor.stats.logS_available} "
        f"({100.0 * processor.stats.logS_available / processor.stats.final_count:.1f}%)"
        if processor.stats.final_count > 0
        else "  logS: 0 (0.0%)"
    )
    print(
        f"  logP: {processor.stats.logP_available} "
        f"({100.0 * processor.stats.logP_available / processor.stats.final_count:.1f}%)"
        if processor.stats.final_count > 0
        else "  logP: 0 (0.0%)"
    )
    print(
        f"  pKa:  {processor.stats.pKa_available} "
        f"({100.0 * processor.stats.pKa_available / processor.stats.final_count:.1f}%)"
        if processor.stats.final_count > 0
        else "  pKa:  0 (0.0%)"
    )
    print(f"  All 3: {processor.stats.all_props_available}")
    print("-" * 60)
    print(f"Output: {args.output}")
    print(f"Report: {args.report}")
    print(f"Time:   {processor.stats.processing_time_sec:.1f}s")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
