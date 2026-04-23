"""Compare GNN ablation checkpoints on an identical held-out test split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, open_dict

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from gnn.evaluation.ablation import (  # noqa: E402
    evaluate_checkpoints_with_eval_config,
    load_prediction_export,
    run_ablation_comparison,
)
from scripts.compare_baselines import parse_model_specs  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=[],
        help=(
            "Checkpoint specs using compare_baselines syntax, e.g. "
            "`AblationA=gnn:/path/model_a.ckpt AblationB=gnn:/path/model_b.ckpt`."
        ),
    )
    parser.add_argument(
        "--prediction-exports",
        nargs="+",
        default=[],
        help=(
            "Optional pre-exported prediction payloads in `name=/path/export.json` format. "
            "When provided, matching models are loaded from these exports instead of re-running eval."
        ),
    )
    parser.add_argument(
        "--confidence-summaries",
        nargs="+",
        default=[],
        help=(
            "Optional sweep summary payloads in `name=/path/multiseed_summary.json` format. "
            "When provided, confidence interval metadata from aggregate_stats is passed through "
            "to comparison CSV/Markdown outputs."
        ),
    )
    parser.add_argument(
        "--config-name", default="eval.yaml", help="Hydra config name for evaluation"
    )
    parser.add_argument(
        "--config-path",
        default="../configs",
        help="Hydra config path for evaluation composition",
    )
    parser.add_argument(
        "--eval-overrides",
        nargs="*",
        default=[],
        help="Optional Hydra override strings applied to eval config",
    )
    parser.add_argument(
        "--output-dir",
        default="reports/ablation_comparison",
        help="Directory for comparison CSV/Markdown/significance artifacts",
    )
    parser.add_argument(
        "--significance-test",
        choices=["wilcoxon", "ttest_rel"],
        default="wilcoxon",
        help="Paired significance test applied to aligned per-example errors",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = _parse_checkpoint_specs(args.models)
    export_paths = _parse_named_paths(args.prediction_exports, label="prediction export")
    confidence_summary_paths = _parse_named_paths(
        args.confidence_summaries,
        label="confidence summary",
    )
    confidence_intervals = _load_confidence_intervals(confidence_summary_paths)
    if not checkpoint_paths and not export_paths:
        raise ValueError(
            "Provide at least one model checkpoint via --models or one export via --prediction-exports"
        )

    models_to_evaluate = {
        model_name: checkpoint_path
        for model_name, checkpoint_path in checkpoint_paths.items()
        if model_name not in export_paths
    }
    if models_to_evaluate:
        eval_cfg = _compose_eval_config(
            config_path=args.config_path,
            config_name=args.config_name,
            overrides=args.eval_overrides,
            output_dir=output_dir,
        )
        evaluated_exports = evaluate_checkpoints_with_eval_config(
            checkpoint_paths=models_to_evaluate,
            eval_cfg=eval_cfg,
            prediction_output_dir=output_dir / "predictions",
        )
        export_paths.update(evaluated_exports)

    exports = {
        model_name: load_prediction_export(path, model_name=model_name)
        for model_name, path in sorted(export_paths.items())
    }
    artifacts = run_ablation_comparison(
        exports,
        output_dir=output_dir,
        significance_test=args.significance_test,
        confidence_intervals_by_model=confidence_intervals,
    )

    logger.info("Saved comparison CSV to %s", artifacts.comparison_csv)
    logger.info("Saved significance CSV to %s", artifacts.significance_csv)
    logger.info("Saved comparison report to %s", artifacts.report_md)


def _parse_checkpoint_specs(model_specs: list[str]) -> dict[str, str]:
    if not model_specs:
        return {}
    parsed_specs = parse_model_specs(model_specs)
    checkpoint_paths: dict[str, str] = {}
    for spec in parsed_specs:
        if spec.source != "artifact":
            raise ValueError(
                f"Model spec {spec.raw!r} is not a checkpoint artifact. "
                "Ablation comparison expects `name=gnn:/path/to.ckpt` entries."
            )
        if spec.kind != "gnn":
            raise ValueError(
                f"Model spec {spec.raw!r} has type {spec.kind!r}. "
                "Ablation comparison currently supports only GNN checkpoints."
            )
        if spec.path is None:
            raise ValueError(f"Model spec {spec.raw!r} is missing an artifact path")
        checkpoint_paths[spec.name] = str(spec.path)
    return checkpoint_paths


def _parse_named_paths(entries: list[str], *, label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for entry in entries:
        name, separator, path_value = entry.partition("=")
        if not separator or not name or not path_value:
            raise ValueError(f"Invalid {label} entry {entry!r}; expected 'name=/path/to/file.json'")
        if name in parsed:
            raise ValueError(f"{label.title()} names must be unique; duplicate entry for {name!r}")
        path = Path(path_value).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"{label.title()} file does not exist: {path}")
        parsed[name] = path
    return parsed


def _load_confidence_intervals(
    summary_paths: dict[str, Path],
) -> dict[str, dict[str, dict[str, object]]]:
    intervals_by_model: dict[str, dict[str, dict[str, object]]] = {}
    for model_name, path in sorted(summary_paths.items()):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(
                f"Confidence summary for model {model_name!r} must be a JSON object: {path}"
            )
        aggregate_stats = payload.get("aggregate_stats")
        if not isinstance(aggregate_stats, dict):
            raise ValueError(
                f"Confidence summary for model {model_name!r} does not contain aggregate_stats: {path}"
            )
        intervals_by_model[model_name] = {
            str(metric_name): dict(summary)
            for metric_name, summary in aggregate_stats.items()
            if isinstance(summary, dict)
        }
    return intervals_by_model


def _compose_eval_config(
    *,
    config_path: str,
    config_name: str,
    overrides: list[str],
    output_dir: Path,
) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path=config_path):
        cfg = compose(
            config_name=config_name,
            return_hydra_config=True,
            overrides=list(overrides),
        )
    with open_dict(cfg):
        cfg.paths.output_dir = str(output_dir)
        cfg.paths.log_dir = str(output_dir)
    return cfg


if __name__ == "__main__":
    main()
