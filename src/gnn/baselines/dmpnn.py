"""Chemprop-backed D-MPNN baseline utilities."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from gnn.baselines.data_utils import BaselineDataset
    from gnn.data.datamodules import PFASBenchDataModule

logger = logging.getLogger(__name__)

DEFAULT_CHEMPROP_ENV_VAR = "MOML_CHEMPROP_COMMAND"
METADATA_FILENAME = "moml_dmpnn_metadata.json"
TRAIN_COLUMN = "smiles"
SPLIT_COLUMN = "split"
DEFAULT_PROPERTY_PREFIX = "target_"


@dataclass
class ChempropDMPNNBaseline:
    """Wrapper around a trained Chemprop D-MPNN artifact."""

    artifact_dir: Path
    property_names: list[str]
    chemprop_command: list[str]
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        self.artifact_dir = Path(self.artifact_dir)
        self.property_names = list(self.property_names)
        self.chemprop_command = list(self.chemprop_command)
        self.metadata_path = self.artifact_dir / METADATA_FILENAME
        self.model_path = self.artifact_dir / "model"
        self.moml_property_names = list(self.property_names)

    def predict(self, smiles: Sequence[str] | np.ndarray) -> np.ndarray:
        """Run Chemprop prediction for one or more SMILES strings."""
        smiles_list = _normalize_smiles(smiles)
        return _predict_with_chemprop(
            smiles=smiles_list,
            property_names=self.property_names,
            artifact_dir=self.artifact_dir,
            chemprop_command=self.chemprop_command,
        )

    def predict_dataset(
        self,
        dataset: BaselineDataset,
        *,
        split_name: str,
    ) -> np.ndarray:
        """Predict a named split from a baseline dataset container."""
        split_smiles = _dataset_split_smiles(dataset, split_name)
        return self.predict(split_smiles)

    def predict_datamodule(
        self,
        datamodule: PFASBenchDataModule,
        *,
        split_name: str,
    ) -> np.ndarray:
        """Predict a named split directly from a PFASBench datamodule."""
        split_smiles = _datamodule_split_smiles(datamodule, split_name)
        return self.predict(split_smiles)

    def save(self, path: str | Path) -> None:
        """Persist the native Chemprop artifacts plus wrapper metadata."""
        save_dmpnn_model(self, path)

    @classmethod
    def load(cls, path: str | Path) -> ChempropDMPNNBaseline:
        """Restore a saved Chemprop D-MPNN artifact wrapper."""
        return load_dmpnn_model(path)

    def get_feature_importances(self) -> pd.DataFrame:
        """D-MPNN baselines do not expose tabular feature importances."""
        raise RuntimeError("Chemprop D-MPNN models do not expose feature importances.")


def train_dmpnn_baseline(
    smiles: Sequence[str] | np.ndarray,
    labels: np.ndarray,
    *,
    smiles_val: Sequence[str] | np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    smiles_test: Sequence[str] | np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    property_names: list[str] | None = None,
    hidden_size: int = 300,
    depth: int = 3,
    epochs: int = 30,
    random_state: int = 42,
    output_dir: str | Path | None = None,
    chemprop_command: str | Sequence[str] | None = None,
    accelerator: str = "cpu",
    devices: int = 1,
    metric: str = "mae",
    batch_size: int = 32,
    **kwargs: Any,
) -> ChempropDMPNNBaseline:
    """Train a Chemprop D-MPNN on SMILES/property pairs.

    Args:
        smiles: Training SMILES strings.
        labels: Training targets with shape ``(n_samples, n_targets)``.
        smiles_val: Optional validation SMILES strings.
        y_val: Optional validation targets aligned with ``smiles_val``.
        smiles_test: Optional test SMILES strings.
        y_test: Optional test targets aligned with ``smiles_test``.
        property_names: Optional names for the target columns.
        hidden_size: Chemprop message hidden dimension.
        depth: Number of message-passing steps.
        epochs: Number of training epochs.
        random_state: Split/training seed for reproducibility.
        output_dir: Optional destination for native Chemprop outputs.
        chemprop_command: Optional command override. Supports a shell string or
            argument sequence. Defaults to ``$MOML_CHEMPROP_COMMAND``, then a
            local ``chemprop`` executable, then ``poetry run chemprop``.
        accelerator: Chemprop accelerator value, defaults to CPU for stable local use.
        devices: Number of devices for Chemprop.
        metric: Validation metric passed to Chemprop.
        batch_size: Chemprop batch size.
        **kwargs: Additional CLI options converted to ``--kebab-case value``.

    Returns:
        A wrapper around the trained Chemprop artifact directory.
    """
    train_smiles = _normalize_smiles(smiles)
    train_labels = _normalize_labels(labels)
    target_names = _resolve_property_names(train_labels.shape[1], property_names)

    if smiles_val is not None or y_val is not None:
        if smiles_val is None or y_val is None:
            raise ValueError("smiles_val and y_val must be provided together")
        val_smiles = _normalize_smiles(smiles_val)
        val_labels = _normalize_labels(y_val, expected_targets=train_labels.shape[1])
    else:
        val_smiles = None
        val_labels = None

    if smiles_test is not None or y_test is not None:
        if smiles_test is None or y_test is None:
            raise ValueError("smiles_test and y_test must be provided together")
        test_smiles = _normalize_smiles(smiles_test)
        test_labels = _normalize_labels(y_test, expected_targets=train_labels.shape[1])
    else:
        test_smiles = None
        test_labels = None

    artifact_dir = (
        Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="moml_dmpnn_"))
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_dir = artifact_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    command_prefix = _resolve_chemprop_command(chemprop_command)
    warmup_epochs = int(kwargs.pop("warmup_epochs", min(2, max(0, epochs - 1))))
    pytorch_seed = int(kwargs.pop("pytorch_seed", random_state))

    train_args: list[str] = []
    if val_smiles is not None and val_labels is not None:
        data_paths = [
            _write_explicit_split_csv(
                artifact_dir / "data" / "splits.csv",
                property_names=target_names,
                train_smiles=train_smiles,
                train_labels=train_labels,
                val_smiles=val_smiles,
                val_labels=val_labels,
                test_smiles=test_smiles,
                test_labels=test_labels,
            )
        ]
        train_args.extend(["--splits-column", SPLIT_COLUMN])
    else:
        data_paths = _write_training_csvs(
            artifact_dir=artifact_dir,
            property_names=target_names,
            train_smiles=train_smiles,
            train_labels=train_labels,
            val_smiles=val_smiles,
            val_labels=val_labels,
            test_smiles=test_smiles,
            test_labels=test_labels,
        )
        if test_smiles is not None and test_labels is not None:
            # Chemprop treats two CSVs as train+test, so force a zero held-out test split.
            train_args.extend(["--split-sizes", "0.9", "0.1", "0.0"])

    command = [
        *command_prefix,
        "train",
        "--data-path",
        *[str(path) for path in data_paths],
        "--task-type",
        "regression",
        "--output-dir",
        str(model_dir),
        "--smiles-columns",
        TRAIN_COLUMN,
        "--target-columns",
        *target_names,
        "--message-hidden-dim",
        str(hidden_size),
        "--depth",
        str(depth),
        "--epochs",
        str(epochs),
        "--warmup-epochs",
        str(warmup_epochs),
        "--metric",
        metric,
        "--batch-size",
        str(batch_size),
        "--accelerator",
        accelerator,
        "--devices",
        str(devices),
        "--data-seed",
        str(random_state),
        "--pytorch-seed",
        str(pytorch_seed),
    ]
    command.extend(train_args)
    command.extend(_cli_args_from_kwargs(kwargs))
    _run_command(command, cwd=artifact_dir)

    metadata = {
        "property_names": target_names,
        "chemprop_command": command_prefix,
        "random_state": random_state,
        "hidden_size": hidden_size,
        "depth": depth,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "metric": metric,
        "pytorch_seed": pytorch_seed,
    }
    (artifact_dir / METADATA_FILENAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return ChempropDMPNNBaseline(
        artifact_dir=artifact_dir,
        property_names=target_names,
        chemprop_command=command_prefix,
    )


def predict_dmpnn(
    model: ChempropDMPNNBaseline,
    smiles: Sequence[str] | np.ndarray,
) -> np.ndarray:
    """Run inference for one or more SMILES strings."""
    return model.predict(smiles)


def save_dmpnn_model(model: ChempropDMPNNBaseline, path: str | Path) -> None:
    """Copy native Chemprop artifacts plus MOML metadata to a new location."""
    output_dir = _resolve_artifact_dir(path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model.artifact_dir, output_dir)


def load_dmpnn_model(path: str | Path) -> ChempropDMPNNBaseline:
    """Load a previously saved Chemprop D-MPNN artifact wrapper."""
    artifact_dir = _resolve_artifact_dir(path)
    metadata_path = artifact_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing D-MPNN metadata: {metadata_path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return ChempropDMPNNBaseline(
        artifact_dir=artifact_dir,
        property_names=list(metadata["property_names"]),
        chemprop_command=list(metadata["chemprop_command"]),
    )


def _predict_with_chemprop(
    *,
    smiles: list[str],
    property_names: list[str],
    artifact_dir: Path,
    chemprop_command: list[str],
) -> np.ndarray:
    input_frame = pd.DataFrame({TRAIN_COLUMN: smiles})

    with tempfile.TemporaryDirectory(prefix="moml_dmpnn_predict_") as tmpdir:
        tmp_path = Path(tmpdir)
        test_path = tmp_path / "predict.csv"
        preds_path = tmp_path / "predictions.csv"
        input_frame.to_csv(test_path, index=False)

        command = [
            *chemprop_command,
            "predict",
            "--test-path",
            str(test_path),
            "--model-paths",
            str(artifact_dir / "model"),
            "--preds-path",
            str(preds_path),
            "--smiles-columns",
            TRAIN_COLUMN,
        ]
        _run_command(command, cwd=artifact_dir)

        predictions = pd.read_csv(preds_path)

    available_columns = set(predictions.columns)
    if set(property_names).issubset(available_columns):
        return predictions[property_names].to_numpy(dtype=float)

    prefixed_columns = [f"pred_{name}" for name in property_names]
    if set(prefixed_columns).issubset(available_columns):
        return predictions[prefixed_columns].to_numpy(dtype=float)

    unknown_columns = [
        column
        for column in predictions.columns
        if column != TRAIN_COLUMN and not column.startswith("uncal_")
    ]
    if len(unknown_columns) >= len(property_names):
        return predictions[unknown_columns[: len(property_names)]].to_numpy(dtype=float)

    raise RuntimeError(
        "Chemprop prediction output did not contain the expected target columns "
        f"for {property_names!r}"
    )


def _dataset_split_smiles(dataset: BaselineDataset, split_name: str) -> list[str]:
    split_map = {
        "train": list(dataset.smiles_train),
        "validation": list(dataset.smiles_val),
        "val": list(dataset.smiles_val),
        "test": list(dataset.smiles_test),
    }
    normalized = _normalize_prediction_split_name(split_name)
    return split_map[normalized]


def _datamodule_split_smiles(datamodule: PFASBenchDataModule, split_name: str) -> list[str]:
    if datamodule.dataset is None:
        raise RuntimeError("DataModule not setup. Call datamodule.setup() first.")
    if datamodule.train_idx is None or datamodule.val_idx is None or datamodule.test_idx is None:
        raise RuntimeError("DataModule split indices are not initialized")

    index_map = {
        "train": datamodule.train_idx,
        "validation": datamodule.val_idx,
        "val": datamodule.val_idx,
        "test": datamodule.test_idx,
    }
    indices = index_map[_normalize_prediction_split_name(split_name)]
    return [datamodule.dataset.get_smiles(int(index)) for index in indices]


def _write_training_csvs(
    *,
    artifact_dir: Path,
    property_names: list[str],
    train_smiles: list[str],
    train_labels: np.ndarray,
    val_smiles: list[str] | None,
    val_labels: np.ndarray | None,
    test_smiles: list[str] | None,
    test_labels: np.ndarray | None,
) -> list[Path]:
    data_dir = artifact_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        _write_split_csv(
            data_dir / "train.csv",
            train_smiles,
            train_labels,
            property_names,
        )
    ]
    if val_smiles is not None and val_labels is not None:
        paths.append(_write_split_csv(data_dir / "val.csv", val_smiles, val_labels, property_names))
    if test_smiles is not None and test_labels is not None:
        paths.append(
            _write_split_csv(data_dir / "test.csv", test_smiles, test_labels, property_names)
        )
    return paths


def _write_explicit_split_csv(
    path: Path,
    *,
    property_names: list[str],
    train_smiles: list[str],
    train_labels: np.ndarray,
    val_smiles: list[str] | None,
    val_labels: np.ndarray | None,
    test_smiles: list[str] | None,
    test_labels: np.ndarray | None,
) -> Path:
    frames = [
        _build_split_frame(
            train_smiles,
            train_labels,
            property_names,
            split_name="train",
        )
    ]
    if val_smiles is not None and val_labels is not None:
        frames.append(
            _build_split_frame(
                val_smiles,
                val_labels,
                property_names,
                split_name="val",
            )
        )
    if test_smiles is not None and test_labels is not None:
        frames.append(
            _build_split_frame(
                test_smiles,
                test_labels,
                property_names,
                split_name="test",
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def _write_split_csv(
    path: Path,
    smiles: list[str],
    labels: np.ndarray,
    property_names: list[str],
) -> Path:
    if len(smiles) != labels.shape[0]:
        raise ValueError(
            "SMILES and labels must contain the same number of rows "
            f"({len(smiles)} != {labels.shape[0]})"
        )

    frame = pd.DataFrame({TRAIN_COLUMN: smiles})
    for index, property_name in enumerate(property_names):
        frame[property_name] = labels[:, index]
    frame.to_csv(path, index=False)
    return path


def _build_split_frame(
    smiles: list[str],
    labels: np.ndarray,
    property_names: list[str],
    *,
    split_name: str,
) -> pd.DataFrame:
    if len(smiles) != labels.shape[0]:
        raise ValueError(
            "SMILES and labels must contain the same number of rows "
            f"({len(smiles)} != {labels.shape[0]})"
        )

    frame = pd.DataFrame({TRAIN_COLUMN: smiles, SPLIT_COLUMN: split_name})
    for index, property_name in enumerate(property_names):
        frame[property_name] = labels[:, index]
    return frame


def _normalize_smiles(smiles: Sequence[str] | np.ndarray) -> list[str]:
    if isinstance(smiles, str):
        normalized = [smiles.strip()]
        if not normalized[0]:
            raise ValueError("SMILES strings must be non-empty")
        return normalized
    flattened = smiles.reshape(-1).tolist() if isinstance(smiles, np.ndarray) else list(smiles)
    normalized = [str(value).strip() for value in flattened]
    if not normalized:
        raise ValueError("At least one SMILES string is required")
    if any(not value for value in normalized):
        raise ValueError("SMILES strings must be non-empty")
    return normalized


def _normalize_labels(
    labels: np.ndarray,
    *,
    expected_targets: int | None = None,
) -> np.ndarray:
    array = np.asarray(labels, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2:
        raise ValueError("labels must be a 1D or 2D array")
    if array.shape[0] == 0:
        raise ValueError("labels must contain at least one row")
    if expected_targets is not None and array.shape[1] != expected_targets:
        raise ValueError(
            "All label matrices must contain the same number of target columns "
            f"({array.shape[1]} != {expected_targets})"
        )
    return array


def _resolve_property_names(num_targets: int, property_names: list[str] | None) -> list[str]:
    if property_names is None:
        return [f"{DEFAULT_PROPERTY_PREFIX}{index}" for index in range(num_targets)]
    if len(property_names) != num_targets:
        raise ValueError(
            "property_names must match the number of target columns "
            f"({len(property_names)} != {num_targets})"
        )
    return list(property_names)


def _resolve_chemprop_command(
    command: str | Sequence[str] | None,
) -> list[str]:
    if command is not None:
        if isinstance(command, str):
            parsed = shlex.split(command)
        else:
            parsed = [str(part) for part in command]
        if parsed:
            return parsed

    env_command = os.environ.get(DEFAULT_CHEMPROP_ENV_VAR)
    if env_command:
        parsed = shlex.split(env_command)
        if parsed:
            return parsed

    if shutil.which("chemprop") is not None:
        return ["chemprop"]

    if shutil.which("poetry") is not None:
        return ["poetry", "run", "chemprop"]

    raise RuntimeError(
        "Unable to locate the Chemprop CLI. Install `chemprop` or set "
        f"{DEFAULT_CHEMPROP_ENV_VAR} to an explicit command."
    )


def _cli_args_from_kwargs(kwargs: dict[str, Any]) -> list[str]:
    arguments: list[str] = []
    for key, value in kwargs.items():
        option = f"--{key.replace('_', '-')}"
        if value is None or value is False:
            continue
        if value is True:
            arguments.append(option)
            continue
        if isinstance(value, (list, tuple)):
            arguments.append(option)
            arguments.extend(str(item) for item in value)
            continue
        arguments.extend([option, str(value)])
    return arguments


def _run_command(command: list[str], *, cwd: Path) -> None:
    logger.info("Running command: %s", shlex.join(command))
    env = os.environ.copy()
    if "MPLCONFIGDIR" not in env:
        mpl_config_dir = cwd / ".mplconfig"
        mpl_config_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(mpl_config_dir)
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        details = stderr or stdout or "no process output captured"
        raise RuntimeError(f"Chemprop command failed: {details}")


def _resolve_artifact_dir(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.suffix == "" else candidate.with_suffix("")


def _normalize_prediction_split_name(split_name: str) -> str:
    if split_name in {"train", "validation", "val", "test"}:
        return split_name
    logger.debug("Unknown split name %s for D-MPNN prediction; defaulting to test", split_name)
    return "test"


__all__ = [
    "ChempropDMPNNBaseline",
    "load_dmpnn_model",
    "predict_dmpnn",
    "save_dmpnn_model",
    "train_dmpnn_baseline",
]
