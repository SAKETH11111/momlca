"""Tests for the Chemprop-backed D-MPNN baseline wrapper."""

from __future__ import annotations

import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from gnn.baselines import ModelComparison
from gnn.baselines.data_utils import BaselineDataset
from gnn.baselines.dmpnn import (
    ChempropDMPNNBaseline,
    load_dmpnn_model,
    predict_dmpnn,
    save_dmpnn_model,
    train_dmpnn_baseline,
)
from scripts import compare_baselines


def _smiles_features(smiles: str) -> list[float]:
    return [
        float(len(smiles)),
        float(smiles.count("F")),
        float(smiles.count("O")),
        float(smiles.count("N")),
        float(smiles.count("C")),
        float(smiles.count("=")),
        float(smiles.count("(") + smiles.count(")")),
    ]


def _labels_from_smiles(smiles_list: list[str]) -> np.ndarray:
    features = np.asarray([_smiles_features(smiles) for smiles in smiles_list], dtype=float)
    return np.column_stack(
        [
            0.12 * features[:, 0] + 0.8 * features[:, 1] - 0.4 * features[:, 2],
            0.35 * features[:, 4] + 0.15 * features[:, 5] - 0.1 * features[:, 6],
            -0.2 * features[:, 0] + 0.5 * features[:, 2] + 0.4 * features[:, 3],
        ]
    )


@pytest.fixture
def fake_chemprop_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Create a tiny executable that mimics the Chemprop CLI contract used by the wrapper."""
    script_path = tmp_path / "fake_chemprop.py"
    script_path.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def featurize(smiles: str) -> list[float]:
    return [
        float(len(smiles)),
        float(smiles.count("F")),
        float(smiles.count("O")),
        float(smiles.count("N")),
        float(smiles.count("C")),
        float(smiles.count("=")),
        float(smiles.count("(") + smiles.count(")")),
    ]


def train(args: argparse.Namespace) -> None:
    train_frame = pd.read_csv(args.data_path[0])
    if getattr(args, "splits_column", None) and args.splits_column in train_frame.columns:
        train_frame = train_frame[train_frame[args.splits_column] == "train"].reset_index(drop=True)
    X = np.asarray([featurize(smiles) for smiles in train_frame[args.smiles_columns]], dtype=float)
    X = np.column_stack([X, np.ones(len(X), dtype=float)])
    y = train_frame[args.target_columns].to_numpy(dtype=float)
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "best.ckpt").write_text("stub checkpoint", encoding="utf-8")
    (output_dir / "stub_model.json").write_text(
        json.dumps({"target_columns": args.target_columns, "coef": coef.tolist()}),
        encoding="utf-8",
    )


def predict(args: argparse.Namespace) -> None:
    model_dir = Path(args.model_paths[0])
    payload = json.loads((model_dir / "stub_model.json").read_text(encoding="utf-8"))
    coef = np.asarray(payload["coef"], dtype=float)
    target_columns = list(payload["target_columns"])

    frame = pd.read_csv(args.test_path)
    X = np.asarray([featurize(smiles) for smiles in frame[args.smiles_columns]], dtype=float)
    X = np.column_stack([X, np.ones(len(X), dtype=float)])
    predictions = X @ coef

    output = pd.DataFrame({args.smiles_columns: frame[args.smiles_columns]})
    for index, column in enumerate(target_columns):
        output[column] = predictions[:, index]
    output.to_csv(args.preds_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data-path", nargs="+", required=True)
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--smiles-columns", required=True)
    train_parser.add_argument("--target-columns", nargs="+", required=True)
    train_parser.add_argument("--task-type")
    train_parser.add_argument("--message-hidden-dim")
    train_parser.add_argument("--depth")
    train_parser.add_argument("--epochs")
    train_parser.add_argument("--metric")
    train_parser.add_argument("--batch-size")
    train_parser.add_argument("--accelerator")
    train_parser.add_argument("--devices")
    train_parser.add_argument("--data-seed")
    train_parser.add_argument("--pytorch-seed")
    train_parser.add_argument("--splits-column")
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--test-path", required=True)
    predict_parser.add_argument("--model-paths", nargs="+", required=True)
    predict_parser.add_argument("--preds-path", required=True)
    predict_parser.add_argument("--smiles-columns", required=True)
    predict_parser.set_defaults(func=predict)

    args, _ = parser.parse_known_args()
    args.func(args)


if __name__ == "__main__":
    main()
""",
        encoding="utf-8",
    )
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("MOML_CHEMPROP_COMMAND", f"{sys.executable} {script_path}")
    return os.environ["MOML_CHEMPROP_COMMAND"]


@pytest.fixture
def real_chemprop_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Resolve the real Chemprop CLI for integration coverage when available."""
    if sys.version_info < (3, 11):
        pytest.skip("Chemprop integration tests require Python 3.11+")

    chemprop_executable = shutil.which("chemprop")
    if chemprop_executable is None:
        pytest.skip("Chemprop CLI is not available in the current test environment")

    monkeypatch.delenv("MOML_CHEMPROP_COMMAND", raising=False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    return chemprop_executable


@pytest.fixture
def dmpnn_dataset() -> tuple[
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
    list[str],
    np.ndarray,
    list[str],
]:
    """Create a deterministic regression problem over valid SMILES strings."""
    train_smiles = [
        "CCO",
        "CCN",
        "CCF",
        "CCCl",
        "CC(=O)O",
        "c1ccccc1",
        "c1ccc(cc1)O",
        "FC(F)(F)C(=O)O",
    ]
    val_smiles = [
        "CCCO",
        "CC(C)O",
        "NCCO",
    ]
    test_smiles = [
        "CCOC",
        "CC(C)N",
        "FC(F)CO",
        "O=C(O)C(F)(F)F",
    ]
    property_names = ["logS", "logP", "pKa"]
    return (
        train_smiles,
        _labels_from_smiles(train_smiles),
        val_smiles,
        _labels_from_smiles(val_smiles),
        test_smiles,
        _labels_from_smiles(test_smiles),
        property_names,
    )


def test_train_predict_and_mae(
    fake_chemprop_command: str,
    dmpnn_dataset: tuple[
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
    ],
    tmp_path: Path,
) -> None:
    del fake_chemprop_command
    train_smiles, y_train, val_smiles, y_val, test_smiles, y_test, property_names = dmpnn_dataset

    model = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        smiles_test=test_smiles,
        y_test=y_test,
        property_names=property_names,
        random_state=42,
        output_dir=tmp_path / "artifact",
    )

    assert isinstance(model, ChempropDMPNNBaseline)
    predictions = predict_dmpnn(model, test_smiles)
    assert predictions.shape == y_test.shape

    mae = np.mean(np.abs(predictions - y_test))
    assert mae < 1e-8


def test_save_load_round_trip(
    fake_chemprop_command: str,
    dmpnn_dataset: tuple[
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
    ],
    tmp_path: Path,
) -> None:
    del fake_chemprop_command
    train_smiles, y_train, val_smiles, y_val, test_smiles, y_test, property_names = dmpnn_dataset

    model = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        property_names=property_names,
        output_dir=tmp_path / "artifact",
    )
    save_path = tmp_path / "saved_model"
    save_dmpnn_model(model, save_path)
    loaded = load_dmpnn_model(save_path)

    np.testing.assert_allclose(
        predict_dmpnn(model, test_smiles), predict_dmpnn(loaded, test_smiles)
    )
    assert loaded.property_names == property_names


def test_random_state_reproducibility(
    fake_chemprop_command: str,
    dmpnn_dataset: tuple[
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
    ],
    tmp_path: Path,
) -> None:
    del fake_chemprop_command
    train_smiles, y_train, val_smiles, y_val, test_smiles, _, property_names = dmpnn_dataset

    model_one = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        property_names=property_names,
        random_state=7,
        output_dir=tmp_path / "artifact_one",
    )
    model_two = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        property_names=property_names,
        random_state=7,
        output_dir=tmp_path / "artifact_two",
    )

    np.testing.assert_allclose(
        predict_dmpnn(model_one, test_smiles),
        predict_dmpnn(model_two, test_smiles),
    )


def test_model_comparison_integration(
    fake_chemprop_command: str,
    dmpnn_dataset: tuple[
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
    ],
    tmp_path: Path,
) -> None:
    del fake_chemprop_command
    train_smiles, y_train, val_smiles, y_val, test_smiles, y_test, property_names = dmpnn_dataset

    model = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        property_names=property_names,
        output_dir=tmp_path / "artifact",
    )

    comparison = ModelComparison(property_names=property_names)
    comparison.add_model("DMPNN", model, model_type="dmpnn")
    dataset = BaselineDataset(
        X_train=np.zeros((len(train_smiles), 1), dtype=float),
        y_train=y_train,
        X_val=np.zeros((len(val_smiles), 1), dtype=float),
        y_val=y_val,
        X_test=np.zeros((len(test_smiles), 1), dtype=float),
        y_test=y_test,
        feature_names=["placeholder"],
        property_names=property_names,
        smiles_train=train_smiles,
        smiles_val=val_smiles,
        smiles_test=test_smiles,
    )

    comparison.evaluate_all_splits({"test": {"dataset": dataset, "targets": y_test}})
    df = comparison.to_dataframe(split_name="test")

    assert "DMPNN" in df.index
    assert df.loc["DMPNN", "model_type"] == "dmpnn"
    assert df.loc["DMPNN", "mae_mean"] < 1e-8


def test_compare_baselines_loads_dmpnn_artifact(
    fake_chemprop_command: str,
    dmpnn_dataset: tuple[
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
    ],
    tmp_path: Path,
) -> None:
    del fake_chemprop_command
    train_smiles, y_train, val_smiles, y_val, test_smiles, _, property_names = dmpnn_dataset

    model = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        property_names=property_names,
        output_dir=tmp_path / "artifact",
    )
    save_path = tmp_path / "saved_dmpnn"
    save_dmpnn_model(model, save_path)

    specs = compare_baselines.parse_model_specs([f"SavedDMPNN=dmpnn:{save_path}"])
    loaded = compare_baselines.load_artifact_models(specs)

    assert "SavedDMPNN" in loaded
    np.testing.assert_allclose(
        predict_dmpnn(model, test_smiles),
        predict_dmpnn(loaded["SavedDMPNN"], test_smiles),
    )




def test_load_does_not_trust_metadata_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_dir = tmp_path / "artifact"
    (artifact_dir / "model").mkdir(parents=True)
    (artifact_dir / "moml_dmpnn_metadata.json").write_text(
        '{"property_names": ["target_0"], "chemprop_command": ["bash", "-c", "echo pwned"]}',
        encoding="utf-8",
    )

    monkeypatch.setenv("MOML_CHEMPROP_COMMAND", "python /safe/chemprop_stub.py")
    model = load_dmpnn_model(artifact_dir)

    assert model.chemprop_command == ["python", "/safe/chemprop_stub.py"]

def test_compare_baselines_defaults_include_dmpnn_when_available() -> None:
    """Default comparison runs should include D-MPNN whenever Chemprop is installed."""
    parsed = compare_baselines.build_parser().parse_args([])
    expected = ["rf", "xgb"]
    if shutil.which("chemprop") is not None:
        expected.append("dmpnn")
    assert parsed.models == expected


@pytest.mark.slow
def test_real_chemprop_cli_supports_explicit_splits_and_reproducibility(
    real_chemprop_command: str,
    dmpnn_dataset: tuple[
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
        np.ndarray,
        list[str],
    ],
    tmp_path: Path,
) -> None:
    train_smiles, y_train, val_smiles, y_val, test_smiles, y_test, property_names = dmpnn_dataset

    model_one = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        smiles_test=test_smiles,
        y_test=y_test,
        property_names=property_names,
        chemprop_command=[real_chemprop_command],
        epochs=1,
        batch_size=4,
        random_state=7,
        output_dir=tmp_path / "real_one",
    )
    model_two = train_dmpnn_baseline(
        train_smiles,
        y_train,
        smiles_val=val_smiles,
        y_val=y_val,
        smiles_test=test_smiles,
        y_test=y_test,
        property_names=property_names,
        chemprop_command=[real_chemprop_command],
        epochs=1,
        batch_size=4,
        random_state=7,
        output_dir=tmp_path / "real_two",
    )

    preds_one = predict_dmpnn(model_one, test_smiles)
    preds_two = predict_dmpnn(model_two, test_smiles)

    assert preds_one.shape == y_test.shape
    np.testing.assert_allclose(preds_one, preds_two)
