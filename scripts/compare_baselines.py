"""Train descriptor baselines and write comparison artifacts."""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gnn.baselines import (
    DescriptorExtractor,
    ModelComparison,
    extract_baseline_data,
    load_rf_model,
    load_xgb_model,
    train_rf_baseline,
    train_xgb_baseline,
)
from gnn.data.datamodules import PFASBenchDataModule

if TYPE_CHECKING:
    from gnn.baselines.data_utils import BaselineDataset

logger = logging.getLogger(__name__)

SPLIT_PRESETS: dict[str, dict[str, object]] = {
    "random": {"split": "random"},
    "scaffold": {"split": "scaffold"},
    "pfas_ood_chain": {
        "split": "pfas_ood",
        "holdout": "chain_length",
        "holdout_values": ["C8"],
    },
    "pfas_ood_headgroup": {
        "split": "pfas_ood",
        "holdout": "headgroup",
        "holdout_values": ["sulfonate"],
    },
    # Backward-compatible aliases for the originally advertised CLI names.
    "ood_chain": {
        "split": "pfas_ood",
        "holdout": "chain_length",
        "holdout_values": ["C8"],
    },
    "ood_headgroup": {
        "split": "pfas_ood",
        "holdout": "headgroup",
        "holdout_values": ["sulfonate"],
    },
}

DISPLAY_NAMES = {
    "rf": "RandomForest",
    "xgb": "XGBoost",
    "gnn": "GNN",
}


class SupportsTabularPredict(Protocol):
    def predict(self, X: Any) -> Any: ...


class SupportsSplitPredict(Protocol):
    def predict_dataset(self, dataset: BaselineDataset, *, split_name: str) -> Any: ...


class SupportsDatamodulePredict(Protocol):
    def predict_datamodule(self, datamodule: PFASBenchDataModule, *, split_name: str) -> Any: ...


@dataclass(frozen=True)
class ModelSpec:
    """CLI model spec parsed from ``--models``."""

    raw: str
    name: str
    kind: str
    source: str
    path: Path | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data", help="PFASBench root directory")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "xgb"],
        help=(
            "Models to compare. Use `rf` / `xgb` to retrain baseline families, or "
            "`name=rf:/path`, `name=xgb:/path`, `name=gnn:/path/to.ckpt` to load "
            "pretrained artifacts."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["scaffold", "random", "pfas_ood_chain", "pfas_ood_headgroup"],
        choices=sorted(SPLIT_PRESETS),
        help="One or more split presets to evaluate",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--descriptor-set", default="physicochemical")
    parser.add_argument(
        "--output",
        default="reports/baseline_comparison.md",
        help="Markdown report output path",
    )
    parser.add_argument(
        "--wandb-mode",
        default="disabled",
        choices=["disabled", "offline", "online"],
        help="Weights & Biases logging mode",
    )
    parser.add_argument(
        "--wandb-project",
        default="moml-baselines",
        help="W&B project name when logging is enabled",
    )
    parser.add_argument(
        "--gnn-loader",
        default=None,
        help=(
            "Import path for loading GNN checkpoints, e.g. "
            "`package.module:load_predictor`. The loader should return an object "
            "with either `predict_datamodule(datamodule, split_name=...)`, "
            "`predict_dataset(dataset, split_name=...)`, or `predict(X)`."
        ),
    )
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--xgb-estimators", type=int, default=500)
    parser.add_argument("--disable-normalization", action="store_true")
    return parser


def parse_model_specs(model_args: list[str]) -> list[ModelSpec]:
    specs = [_parse_model_spec(arg) for arg in model_args]
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("Model names must be unique across --models entries")
    return specs


def _parse_model_spec(value: str) -> ModelSpec:
    if value in {"rf", "xgb"}:
        return ModelSpec(
            raw=value,
            name=DISPLAY_NAMES[value],
            kind=value,
            source="train",
        )

    if "=" not in value:
        raise ValueError("Model specs must be `rf`, `xgb`, or `name=<rf|xgb|gnn>:<path>` entries")

    name, remainder = value.split("=", maxsplit=1)
    if not name:
        raise ValueError(f"Invalid model spec {value!r}: missing model name before `=`")
    if ":" not in remainder:
        raise ValueError(
            f"Invalid model spec {value!r}: expected `<kind>:<path>` after the model name"
        )

    kind, path_value = remainder.split(":", maxsplit=1)
    if kind not in {"rf", "xgb", "gnn"}:
        raise ValueError(f"Invalid model type {kind!r} in {value!r}; expected one of rf, xgb, gnn")
    path = Path(path_value).expanduser()
    return ModelSpec(raw=value, name=name, kind=kind, source="artifact", path=path)


def load_artifact_models(
    model_specs: list[ModelSpec],
    *,
    gnn_loader: str | None = None,
) -> dict[str, Any]:
    loaded_models: dict[str, Any] = {}
    gnn_loader_callable = None if gnn_loader is None else _import_callable(gnn_loader)

    for spec in model_specs:
        if spec.source != "artifact":
            continue
        assert spec.path is not None
        if not spec.path.exists():
            raise FileNotFoundError(f"Model artifact does not exist: {spec.path}")

        if spec.kind == "rf":
            loaded_models[spec.name] = load_rf_model(spec.path)
        elif spec.kind == "xgb":
            loaded_models[spec.name] = load_xgb_model(spec.path)
        else:
            if gnn_loader_callable is None:
                raise ValueError("GNN model specs require --gnn-loader package.module:callable")
            loaded_models[spec.name] = _call_loader(gnn_loader_callable, spec.path)

    return loaded_models


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    model_specs = parse_model_specs(args.models)
    loaded_models = load_artifact_models(model_specs, gnn_loader=args.gnn_loader)

    report_path = Path(args.output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = report_path.with_suffix(".csv")

    comparison: ModelComparison | None = None
    for split_alias in args.splits:
        datamodule = _build_datamodule(split_alias, data_root=args.data_root, seed=args.seed)
        datamodule.setup()

        dataset = extract_baseline_data(
            datamodule,
            extractor=DescriptorExtractor(descriptor_set=args.descriptor_set),
            normalize_features=not args.disable_normalization,
        )
        if comparison is None:
            comparison = ModelComparison(
                property_names=dataset.property_names,
                use_wandb=args.wandb_mode != "disabled",
            )

        for spec in model_specs:
            model = (
                _train_model_for_split(spec, dataset, seed=args.seed, args=args)
                if spec.source == "train"
                else loaded_models[spec.name]
            )
            predictions = predict_model(
                model,
                dataset=dataset,
                datamodule=datamodule,
                split_name=split_alias,
            )
            comparison.add_result(
                spec.name,
                predictions,
                dataset.y_test,
                split_name=split_alias,
                metadata=_model_metadata(spec, args),
            )

    if comparison is None:
        raise RuntimeError("No comparison results were generated")

    comparison.save(csv_path)
    comparison.save_report(report_path)

    run = _maybe_start_wandb(
        mode=args.wandb_mode,
        project=args.wandb_project,
        report_path=report_path,
        models=[spec.raw for spec in model_specs],
        splits=args.splits,
    )
    try:
        comparison.log_to_wandb(run=run)
    finally:
        if run is not None:
            run.finish()

    logger.info("Saved comparison CSV to %s", csv_path)
    logger.info("Saved comparison report to %s", report_path)
    logger.info(
        "\n%s",
        comparison.to_table(metric_types=["mae", "rmse", "r2", "spearman"]),
    )


def _build_datamodule(split_alias: str, *, data_root: str, seed: int) -> PFASBenchDataModule:
    split_config = SPLIT_PRESETS[split_alias]
    return PFASBenchDataModule(root=data_root, seed=seed, **split_config)


def _train_model_for_split(
    spec: ModelSpec,
    dataset: BaselineDataset,
    *,
    seed: int,
    args: argparse.Namespace,
) -> Any:
    if spec.kind == "rf":
        return train_rf_baseline(
            dataset.X_train,
            dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            property_names=dataset.property_names,
            feature_names=dataset.feature_names,
            n_estimators=args.rf_estimators,
            random_state=seed,
        )

    if spec.kind == "xgb":
        return train_xgb_baseline(
            dataset.X_train,
            dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            property_names=dataset.property_names,
            feature_names=dataset.feature_names,
            n_estimators=args.xgb_estimators,
            random_state=seed,
        )

    raise ValueError(f"Unsupported trainable model type: {spec.kind}")


def predict_model(
    model: Any,
    *,
    dataset: BaselineDataset,
    datamodule: PFASBenchDataModule | None = None,
    split_name: str,
) -> np.ndarray:
    if hasattr(model, "predict_datamodule"):
        if datamodule is None:
            raise ValueError(
                "predict_datamodule(...) models require the split datamodule to be provided"
            )
        prediction = model.predict_datamodule(datamodule, split_name=split_name)
        return np.asarray(prediction, dtype=float)

    if hasattr(model, "predict_dataset"):
        prediction = model.predict_dataset(dataset, split_name=split_name)
        return np.asarray(prediction, dtype=float)

    if hasattr(model, "predict"):
        prediction = model.predict(dataset.X_test)
        return np.asarray(prediction, dtype=float)

    raise TypeError(
        "Loaded model must expose `predict_datamodule(datamodule, split_name=...)`, "
        "`predict_dataset(dataset, split_name=...)`, or `predict(X)`"
    )


def _model_metadata(spec: ModelSpec, args: argparse.Namespace) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_type": spec.kind,
        "source": spec.source,
    }
    if spec.kind == "rf" and spec.source == "train":
        metadata["n_estimators"] = args.rf_estimators
    elif spec.kind == "xgb" and spec.source == "train":
        metadata["n_estimators"] = args.xgb_estimators
    elif spec.path is not None:
        metadata["artifact_path"] = str(spec.path)
    return metadata


def _import_callable(import_path: str) -> Any:
    module_name, separator, attr_name = import_path.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError(
            f"Invalid callable import path {import_path!r}; expected package.module:callable"
        )
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _call_loader(loader: Any, path: Path) -> Any:
    signature = inspect.signature(loader)
    if len(signature.parameters) == 1:
        return loader(path)
    return loader(path=path)


def _maybe_start_wandb(
    *,
    mode: str,
    project: str,
    report_path: Path,
    models: list[str],
    splits: list[str],
):
    if mode == "disabled":
        return None

    import wandb

    return wandb.init(
        project=project,
        mode=mode,
        job_type="baseline-comparison",
        config={
            "report_path": str(report_path),
            "models": models,
            "splits": splits,
        },
    )


if __name__ == "__main__":
    main()
