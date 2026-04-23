"""Run PFAS family error analysis from a checkpoint or existing prediction export."""

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

from gnn.evaluation.ablation import evaluate_checkpoints_with_eval_config  # noqa: E402
from gnn.evaluation.family_analysis import (  # noqa: E402
    load_family_analysis_input,
    run_family_error_analysis,
)
from scripts.compare_baselines import parse_model_specs  # noqa: E402

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for family analysis orchestration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional checkpoint spec using compare_baselines syntax, e.g. "
            "`FamilyEval=gnn:/path/to/model.ckpt`."
        ),
    )
    parser.add_argument(
        "--prediction-export",
        default=None,
        help="Optional existing prediction export JSON path from canonical eval export.",
    )
    parser.add_argument("--config-name", default="eval.yaml", help="Hydra eval config name")
    parser.add_argument(
        "--config-path",
        default="../configs",
        help="Hydra eval config path for composition",
    )
    parser.add_argument(
        "--eval-overrides",
        nargs="*",
        default=[],
        help=(
            "Optional Hydra overrides when evaluating from --model, e.g. "
            "`data=pfasbench_ood_chain trainer=cpu`."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="reports/family_analysis",
        help="Directory for family CSV/Markdown/figure artifacts",
    )
    parser.add_argument(
        "--low-sample-threshold",
        type=int,
        default=5,
        help="Sample-count threshold used to flag low-sample family rows in reports.",
    )
    parser.add_argument(
        "--confidence-summary",
        default=None,
        help=(
            "Optional multiseed_summary.json path. When provided, aggregate confidence "
            "interval stats are copied as a deterministic sidecar artifact."
        ),
    )
    return parser


def main() -> None:
    """Run one deterministic PFAS family-analysis workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args()

    _validate_inputs(model_spec=args.model, prediction_export=args.prediction_export)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    confidence_summary_path = (
        _validate_confidence_summary_path(args.confidence_summary)
        if args.confidence_summary is not None
        else None
    )

    if args.prediction_export is not None:
        export_path = Path(args.prediction_export).expanduser()
        if not export_path.exists():
            raise FileNotFoundError(f"Prediction export does not exist: {export_path}")
    else:
        checkpoint_path = _parse_checkpoint_spec(args.model)
        eval_cfg = _compose_eval_config(
            config_path=args.config_path,
            config_name=args.config_name,
            overrides=list(args.eval_overrides),
            output_dir=output_dir,
        )
        exported = evaluate_checkpoints_with_eval_config(
            checkpoint_paths={"family_analysis": checkpoint_path},
            eval_cfg=eval_cfg,
            prediction_output_dir=output_dir / "predictions",
        )
        export_path = exported["family_analysis"]

    analysis_input = load_family_analysis_input(export_path)
    artifacts = run_family_error_analysis(
        analysis_input=analysis_input,
        output_dir=output_dir,
        low_sample_threshold=args.low_sample_threshold,
    )

    logger.info("Family analysis report: %s", artifacts.report_md)
    logger.info("Chain-length metrics CSV: %s", artifacts.chain_length_csv)
    logger.info("Headgroup metrics CSV: %s", artifacts.headgroup_csv)
    logger.info("Chain-length figure: %s", artifacts.chain_length_figure)
    logger.info("Headgroup figure: %s", artifacts.headgroup_figure)
    if confidence_summary_path is not None:
        sidecar_path = _write_confidence_summary_sidecar(
            source_path=confidence_summary_path,
            output_dir=output_dir,
            report_path=artifacts.report_md,
        )
        logger.info("Confidence summary sidecar: %s", sidecar_path)


def _validate_inputs(*, model_spec: str | None, prediction_export: str | None) -> None:
    if (model_spec is None) == (prediction_export is None):
        raise ValueError("Provide exactly one of --model or --prediction-export")


def _validate_confidence_summary_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Confidence summary does not exist: {path}")
    return path


def _write_confidence_summary_sidecar(
    *,
    source_path: Path,
    output_dir: Path,
    report_path: Path,
) -> Path:
    try:
        payload = json.loads(source_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Confidence summary {source_path} is not valid JSON: {exc.msg} "
            f"(line {exc.lineno}, column {exc.colno})"
        ) from exc
    aggregate_stats = payload.get("aggregate_stats")
    if not isinstance(aggregate_stats, dict):
        raise ValueError(
            f"Confidence summary {source_path} does not contain aggregate_stats mapping"
        )
    sidecar_path = output_dir / f"{report_path.stem}-confidence-summary.json"
    sidecar_path.write_text(
        json.dumps(
            {"source_path": str(source_path), "aggregate_stats": aggregate_stats},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return sidecar_path


def _parse_checkpoint_spec(model_spec: str | None) -> str:
    if model_spec is None:
        raise ValueError("model_spec cannot be None")
    parsed = parse_model_specs([model_spec])
    spec = parsed[0]
    if spec.source != "artifact":
        raise ValueError("Family analysis checkpoint input must be an artifact model spec")
    if spec.kind != "gnn":
        raise ValueError("Family analysis supports only `gnn` checkpoint specs")
    if spec.path is None:
        raise ValueError("Checkpoint path is missing in model spec")
    return str(spec.path)


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
