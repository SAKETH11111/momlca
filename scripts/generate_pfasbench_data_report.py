#!/usr/bin/env python3
"""Generate a DataValidator report for the curated PFASBench CSV.

This avoids the heavyweight PyG dataset processing step by validating directly
from the CSV columns (inchikey + labels), while still using the shared
DataValidator + report generator.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch
from torch_geometric.data import Data

from gnn.data.validation import DataValidator, generate_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a PFASBench data quality report")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/pfasbench/raw/pfasbench.csv"),
        help="Input PFASBench CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/pfasbench/raw/data_report.md"),
        help="Output markdown report path",
    )
    parser.add_argument(
        "--mode",
        choices=["warn", "fail"],
        default="warn",
        help="Validator mode: warn logs issues; fail raises on validation failure",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=3.0,
        help="Z-score threshold for outlier detection",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("gnn.data.validation").setLevel(logging.ERROR)

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    rows: list[dict[str, str]] = []
    with args.input.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    class CsvDataset:
        property_names = ["logS", "logP", "pKa"]
        property_units = {"logS": "log(mol/L)", "logP": "unitless", "pKa": "pH"}

        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, idx: int) -> Data:
            row = rows[idx]
            item = Data()
            item.inchikey = (row.get("inchikey") or "").strip()
            y = []
            for prop in self.property_names:
                raw = (row.get(prop) or "").strip()
                y.append(float("nan") if not raw else float(raw))
            item.y = torch.tensor([y], dtype=torch.float32)
            return item

    dataset = CsvDataset()
    validator = DataValidator(mode=args.mode, outlier_threshold=args.outlier_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_report(dataset, validator=validator, output_path=str(args.output))
    print(f"Wrote report: {args.output}")


if __name__ == "__main__":
    main()
