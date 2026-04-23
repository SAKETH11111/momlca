import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf, open_dict

from gnn.evaluation.confidence_intervals import interval_report_fields
from src.utils import pylogger

log = pylogger.RankedLogger(__name__, rank_zero_only=True)


def _parameter_counts(model: Any) -> tuple[int, int, int]:
    """Count parameters; skip uninitialized / lazy tensors that cannot report numel yet."""
    total = trainable = non_trainable = 0
    for p in model.parameters():
        try:
            n = int(p.numel())
        except ValueError:
            continue
        total += n
        if p.requires_grad:
            trainable += n
        else:
            non_trainable += n
    return total, trainable, non_trainable


def _git_metadata(repo_root: Path | None) -> tuple[str, str]:
    """Return (full_sha, short_sha) for HEAD, or ("", "") if unavailable."""
    cwd = repo_root if repo_root is not None else Path(".")
    try:
        full = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "", ""
    short = full[:7] if len(full) >= 7 else full
    return full, short


def _git_tag(short_sha: str) -> str:
    """Return the canonical W&B tag for a git revision."""
    return f"git:{short_sha}" if short_sha else ""


def _maybe_add_wandb_git_tag(logger: Any, git_short: str) -> None:
    """Ensure W&B runs get a stable git tag for experiment filtering/comparison."""
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        return

    if not git_short or not isinstance(logger, WandbLogger):
        return

    git_tag = _git_tag(git_short)
    wandb_init = getattr(logger, "_wandb_init", None)
    if isinstance(wandb_init, dict):
        tags = list(wandb_init.get("tags") or [])
        if git_tag not in tags:
            wandb_init["tags"] = [*tags, git_tag]


def _maybe_wandb_log_code(
    trainer: Any,
    cfg: Any,
    repo_root: Path | None,
) -> None:
    """Optionally call wandb Run.log_code when W&B is active and config enables it."""
    try:
        from lightning.pytorch.loggers import WandbLogger
    except ImportError:
        return

    wandb_log_code = bool(OmegaConf.select(cfg, "logger.wandb_log_code", default=False))
    if not wandb_log_code or not repo_root:
        return

    for lg in getattr(trainer, "loggers", []) or []:
        if not isinstance(lg, WandbLogger):
            continue
        run = getattr(lg, "experiment", None)
        if run is None or not hasattr(run, "log_code"):
            continue
        try:
            run.log_code(str(repo_root))
        except Exception:
            log.exception("wandb log_code failed; continuing training.")


def apply_wandb_multirun_metadata(
    cfg: Any,
    *,
    group_name: str,
    job_type: str = "multiseed-child",
) -> None:
    """Inject deterministic group metadata into the configured W&B logger."""
    wandb_cfg = OmegaConf.select(cfg, "logger.wandb", default=None)
    if wandb_cfg is None:
        return

    with open_dict(cfg):
        if not OmegaConf.select(cfg, "logger.wandb.group", default=""):
            cfg.logger.wandb.group = group_name
        if not OmegaConf.select(cfg, "logger.wandb.job_type", default=""):
            cfg.logger.wandb.job_type = job_type


def log_multiseed_summary_to_wandb(
    loggers: Sequence[Any],
    summary_payload: Mapping[str, Any],
    *,
    prefix: str = "multiseed",
) -> None:
    """Log sweep-level multi-seed summaries through any configured W&B logger."""
    if not loggers:
        return

    per_run_rows = summary_payload.get("runs") or []
    aggregate_stats = summary_payload.get("aggregate_stats") or {}
    artifacts = summary_payload.get("artifacts") or {}

    for logger in loggers:
        wandb_init = getattr(logger, "_wandb_init", None)
        experiment = getattr(logger, "experiment", None)
        summary = getattr(experiment, "summary", None) if experiment is not None else None
        if not isinstance(wandb_init, dict) or experiment is None or summary is None:
            continue

        summary[f"{prefix}/group"] = summary_payload.get("group_name", "")
        summary[f"{prefix}/run_count"] = int(summary_payload.get("run_count", 0))
        for artifact_name, artifact_path in artifacts.items():
            summary[f"{prefix}/artifacts/{artifact_name}"] = artifact_path
        for metric_name, stats in aggregate_stats.items():
            for stat_name, value in stats.items():
                if value is not None:
                    summary[f"{prefix}/{metric_name}/{stat_name}"] = value

        try:
            import wandb

            if per_run_rows and hasattr(experiment, "log"):
                per_run_columns = [
                    "job_num",
                    "seed",
                    "data_seed",
                    "checkpoint_path",
                    "metrics",
                ]
                per_run_data = [
                    [
                        row.get("job_num", ""),
                        row.get("seed"),
                        row.get("data_seed"),
                        row.get("checkpoint_path", ""),
                        json_safe_metrics(row.get("metrics") or {}),
                    ]
                    for row in per_run_rows
                ]
                aggregate_columns = [
                    "metric",
                    "n",
                    "mean",
                    "std",
                    "sem",
                    "ci_method",
                    "ci_level",
                    "ci_low",
                    "ci_high",
                    "ci_half_width",
                    "ci95",
                    "ci_display",
                ]
                aggregate_data = [
                    [
                        metric_name,
                        interval_fields.get("n"),
                        interval_fields.get("mean"),
                        interval_fields.get("std"),
                        interval_fields.get("sem"),
                        interval_fields.get("ci_method"),
                        interval_fields.get("ci_level"),
                        interval_fields.get("ci_low"),
                        interval_fields.get("ci_high"),
                        interval_fields.get("ci_half_width"),
                        interval_fields.get("ci95"),
                        interval_fields.get("ci_display"),
                    ]
                    for metric_name, stats in aggregate_stats.items()
                    for interval_fields in [interval_report_fields(stats)]
                ]
                experiment.log(
                    {
                        f"{prefix}/per_run": wandb.Table(
                            columns=per_run_columns, data=per_run_data
                        ),
                        f"{prefix}/aggregate": wandb.Table(
                            columns=aggregate_columns, data=aggregate_data
                        ),
                    }
                )
        except Exception:
            log.exception("Failed to log multi-seed W&B summary; continuing training.")


def json_safe_metrics(metrics: Mapping[str, Any]) -> str:
    """Render metric key/value pairs for compact W&B table display."""
    pairs = [f"{metric}={value:.4f}" for metric, value in sorted(metrics.items())]
    return ", ".join(pairs)


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any]) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters
        - Git commit metadata and canonical checkpoint directory for reproducibility

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams: dict[str, Any] = {}

    try:
        cfg = OmegaConf.to_container(object_dict["cfg"], resolve=True)
    except Exception:
        log.warning(
            "Config could not be fully resolved for hyperparameter logging; "
            "logging best-effort unresolved values."
        )
        cfg = OmegaConf.to_container(object_dict["cfg"], resolve=False)
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    loggers = getattr(trainer, "loggers", None) or []
    if not loggers:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters (skip lazy/uninitialized tensors)
    p_total, p_train, p_non = _parameter_counts(model)
    hparams["model/params/total"] = p_total
    hparams["model/params/trainable"] = p_train
    hparams["model/params/non_trainable"] = p_non

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")

    paths = cfg.get("paths") or {}
    output_dir = paths.get("output_dir", "")
    checkpoint_dir = str(Path(str(output_dir)) / "checkpoints") if output_dir else ""
    hparams["checkpoint_output_dir"] = checkpoint_dir
    hparams["hydra_output_dir"] = output_dir

    raw_cfg = object_dict["cfg"]
    root_dir = OmegaConf.select(raw_cfg, "paths.root_dir", default="")
    repo_root = Path(str(root_dir)) if root_dir else None
    git_full, git_short = _git_metadata(repo_root)
    hparams["git_commit"] = git_full
    hparams["git_commit_short"] = git_short
    hparams["git_tag"] = _git_tag(git_short)

    # send hparams to all loggers
    for logger in loggers:
        _maybe_add_wandb_git_tag(logger, git_short)
        logger.log_hyperparams(hparams)

    _maybe_wandb_log_code(trainer, raw_cfg, repo_root)
