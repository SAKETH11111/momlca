"""Train descriptor baselines and write comparison artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gnn.baselines import (
    DescriptorExtractor,
    ModelComparison,
    extract_baseline_data,
    train_rf_baseline,
    train_xgboost_baseline,
)
from gnn.data.datamodules import PFASBenchDataModule


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="PFASBench root directory")
    parser.add_argument(
        "--split",
        default="scaffold",
        choices=["random", "scaffold", "ood_chain", "ood_headgroup"],
        help="PFASBench split to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--descriptor-set", default="physicochemical")
    parser.add_argument("--output-dir", default="outputs/baselines")
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--xgb-estimators", type=int, default=500)
    parser.add_argument("--disable-normalization", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datamodule = PFASBenchDataModule(root=args.data_root, split=args.split, seed=args.seed)
    datamodule.setup()

    dataset = extract_baseline_data(
        datamodule,
        extractor=DescriptorExtractor(descriptor_set=args.descriptor_set),
        normalize_features=not args.disable_normalization,
    )

    rf_model = train_rf_baseline(
        dataset.X_train,
        dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        property_names=dataset.property_names,
        feature_names=dataset.feature_names,
        n_estimators=args.rf_estimators,
        random_state=args.seed,
    )
    xgb_model = train_xgboost_baseline(
        dataset.X_train,
        dataset.y_train,
        X_val=dataset.X_val,
        y_val=dataset.y_val,
        property_names=dataset.property_names,
        feature_names=dataset.feature_names,
        n_estimators=args.xgb_estimators,
        random_state=args.seed,
    )

    comparison = ModelComparison(property_names=dataset.property_names)
    comparison.add_model("RandomForest", rf_model, metadata={"n_estimators": args.rf_estimators})
    comparison.add_model("XGBoost", xgb_model, metadata={"n_estimators": args.xgb_estimators})
    comparison.evaluate_splits(
        {
            "validation": (dataset.X_val, dataset.y_val),
            "test": (dataset.X_test, dataset.y_test),
        }
    )

    comparison.save(output_dir / "comparison.csv")
    comparison.save_report(output_dir / "comparison.md")
    rf_model.save(output_dir / "random_forest.joblib")
    xgb_model.save(output_dir / "xgboost")

    print(
        comparison.to_table(
            metric_types=["mae", "rmse", "r2", "spearman"],
            split_name="test",
        )
    )


if __name__ == "__main__":
    main()
