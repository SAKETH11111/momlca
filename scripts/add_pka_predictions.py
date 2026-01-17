#!/usr/bin/env python3
"""Add pKa predictions to PFASBench dataset using pka-predictor-moitessier.

This script predicts pKa values for all molecules in PFASBench and updates
the dataset with the predictions. By default, it only fills missing pKa values.

Usage:
    python scripts/add_pka_predictions.py
    python scripts/add_pka_predictions.py --input data/pfasbench/raw/pfasbench.csv --output data/pfasbench/raw/pfasbench.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add pKa predictions to PFASBench dataset")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/pfasbench/raw/pfasbench.csv"),
        help="Input PFASBench CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pfasbench/raw/pfasbench.csv"),
        help="Output CSV file (default: overwrite input)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for pKa prediction",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        default=False,
        help="Overwrite existing pKa values (NOT recommended)",
    )
    parser.add_argument(
        "--update-curation-report",
        type=Path,
        default=None,
        help=(
            "Optional path to a curation report JSON file (e.g., "
            "data/pfasbench/raw/curation_report.json) to update property coverage stats after adding pKa"
        ),
    )
    parser.add_argument(
        "--refresh-report-only",
        action="store_true",
        default=False,
        help="Skip prediction and only refresh the curation report stats from the current CSV content",
    )
    return parser.parse_args(argv)


def _import_pka_predict() -> object:
    argv_backup = list(sys.argv)
    sys.argv = [sys.argv[0]]
    try:
        from pka_predictor.predict import predict  # type: ignore

        return predict
    finally:
        sys.argv = argv_backup


def predict_pka_batch(smiles_list: list[str], batch_size: int = 100) -> dict[str, float]:
    """Predict pKa values for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        batch_size: Number of molecules to process at once

    Returns:
        Dictionary mapping SMILES to pKa values (NaN if prediction failed)
    """
    predict = _import_pka_predict()

    results: dict[str, float] = {}

    # Process in batches
    for i in tqdm(range(0, len(smiles_list), batch_size), desc="Predicting pKa"):
        batch = smiles_list[i : i + batch_size]
        df_batch = pd.DataFrame({"smiles": batch})

        try:
            pka_results = predict(df_batch, verbose=0)  # type: ignore[misc]

            # Map results back to input SMILES
            # The predictor may reorder or filter molecules
            for _, row in pka_results.iterrows():
                mol_num = row.get("mol_number", -1)
                if 0 <= mol_num - 1 < len(batch):
                    original_smiles = batch[mol_num - 1]
                    pka = row.get("predicted_pKa", float("nan"))
                    if pd.notna(pka):
                        results[original_smiles] = float(pka)
        except Exception as e:
            logger.warning("Batch %d failed: %s", i // batch_size, e)
            continue

    return results


def add_pka_to_dataset(
    input_csv: Path,
    output_csv: Path,
    batch_size: int = 100,
    overwrite_existing: bool = False,
    update_curation_report: Path | None = None,
) -> None:
    """Add pKa predictions to PFASBench dataset.

    Args:
        input_csv: Path to input PFASBench CSV
        output_csv: Path to output CSV with pKa predictions
        batch_size: Batch size for pKa prediction
        overwrite_existing: If True, overwrite existing pKa values. If False,
            only fill missing values.
        update_curation_report: Optional curation report JSON path to update
            after adding pKa predictions.
    """
    logger.info(f"Reading dataset from {input_csv}")
    df = pd.read_csv(input_csv)
    logger.info(f"Dataset has {len(df)} molecules")

    if "pKa" not in df.columns:
        df["pKa"] = pd.NA
    if "source" not in df.columns:
        df["source"] = ""

    # Get unique SMILES to avoid redundant predictions
    if overwrite_existing:
        smiles_to_predict = df["smiles"].unique().tolist()
        logger.info("Predicting pKa for %d unique SMILES (overwrite mode)", len(smiles_to_predict))
    else:
        missing_mask = df["pKa"].isna()
        smiles_to_predict = df.loc[missing_mask, "smiles"].dropna().unique().tolist()
        logger.info("Predicting pKa for %d unique SMILES with missing pKa", len(smiles_to_predict))

    # Predict pKa values
    if smiles_to_predict:
        pka_map = predict_pka_batch(smiles_to_predict, batch_size=batch_size)
        logger.info("Got pKa predictions for %d molecules", len(pka_map))
    else:
        pka_map = {}
        logger.info("No missing pKa values detected; skipping prediction")

    predicted_series = df["smiles"].map(pka_map)
    if overwrite_existing:
        df["pKa"] = predicted_series
        filled_mask = predicted_series.notna()
    else:
        df.loc[missing_mask, "pKa"] = predicted_series[missing_mask]
        filled_mask = missing_mask & predicted_series.notna()

    # Update source column to indicate pKa source
    tag = "pKa:pka-predictor-moitessier"
    sources = df["source"].fillna("").astype(str)
    needs_tag = filled_mask & ~sources.str.contains("pKa:", regex=False)
    df.loc[needs_tag & (sources != ""), "source"] = sources[needs_tag & (sources != "")] + f";{tag}"
    df.loc[needs_tag & (sources == ""), "source"] = tag

    # Save
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved dataset with pKa to {output_csv}")

    # Report coverage
    n_total = len(df)
    n_pka = df["pKa"].notna().sum()
    n_logp = df["logP"].notna().sum()
    n_logs = df["logS"].notna().sum()

    logger.info("")
    logger.info("Property coverage:")
    logger.info(f"  logP: {n_logp}/{n_total} ({100 * n_logp / n_total:.1f}%)")
    logger.info(f"  logS: {n_logs}/{n_total} ({100 * n_logs / n_total:.1f}%)")
    logger.info(f"  pKa:  {n_pka}/{n_total} ({100 * n_pka / n_total:.1f}%)")

    if update_curation_report is not None and update_curation_report.exists():
        _update_curation_report(update_curation_report, df)


def _update_curation_report(report_path: Path, df: pd.DataFrame) -> None:
    try:
        report = json.loads(report_path.read_text())
    except Exception as e:
        logger.warning("Failed to read curation report %s: %s", report_path, e)
        return

    def _count_nonempty(col: str) -> int:
        if col not in df.columns:
            return 0
        return int(df[col].notna().sum())

    final_count = int(len(df))
    logS_available = _count_nonempty("logS")
    logP_available = _count_nonempty("logP")
    pKa_available = _count_nonempty("pKa")
    all_properties = (
        int(df[["logS", "logP", "pKa"]].notna().all(axis=1).sum()) if final_count else 0
    )

    report.setdefault("output", {})
    report["output"]["final_count"] = final_count
    report["output"].setdefault("properties", {})
    report["output"]["properties"].update(
        {
            "logS_available": logS_available,
            "logP_available": logP_available,
            "pKa_available": pKa_available,
            "all_properties": all_properties,
        }
    )
    report["output"].setdefault("property_coverage", {})
    report["output"]["property_coverage"].update(
        {
            "logS_pct": 100.0 * logS_available / final_count if final_count else 0.0,
            "logP_pct": 100.0 * logP_available / final_count if final_count else 0.0,
            "pKa_pct": 100.0 * pKa_available / final_count if final_count else 0.0,
        }
    )
    report["pka_enrichment"] = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "pKa_added": int(df["pKa"].notna().sum()) if "pKa" in df.columns else 0,
    }

    try:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        logger.info("Updated curation report %s", report_path)
    except Exception as e:
        logger.warning("Failed to write curation report %s: %s", report_path, e)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        raise SystemExit(1)

    if args.refresh_report_only:
        if args.update_curation_report is None:
            logger.error("--refresh-report-only requires --update-curation-report")
            raise SystemExit(2)
        df = pd.read_csv(args.input)
        _update_curation_report(args.update_curation_report, df)
        return

    add_pka_to_dataset(
        args.input,
        args.output,
        args.batch_size,
        overwrite_existing=args.overwrite_existing,
        update_curation_report=args.update_curation_report,
    )


if __name__ == "__main__":
    main()
