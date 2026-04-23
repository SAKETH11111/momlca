"""CLI smoke test for PFAS family error analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_export(path: Path, *, checkpoint_path: str = "/tmp/models/family.ckpt") -> Path:
    payload = {
        "metadata": {
            "split": "test",
            "checkpoint_path": checkpoint_path,
            "property_names": ["logS"],
            "num_records": 1,
        },
        "records": [
            {
                "split": "test",
                "checkpoint_path": checkpoint_path,
                "smiles": "C(=O)(C(F)(F)F)O",
                "name": "TFA",
                "inchikey": "DTZQGRPZKCHYJP-UHFFFAOYSA-N",
                "targets": {"logS": 1.0},
                "predictions": {"logS": 0.9},
                "residuals": {"logS": -0.1},
            }
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def test_analyze_family_errors_cli_with_prediction_export(tmp_path: Path) -> None:
    export_path = _write_export(tmp_path / "predictions.json")
    output_dir = tmp_path / "analysis"
    script_path = Path("scripts/analyze_family_errors.py")

    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--prediction-export",
            str(export_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert any(output_dir.glob("*-family-report.md"))
    assert any(output_dir.glob("*-family-chain_length.csv"))


def test_analyze_family_errors_cli_with_model_path_uses_eval_export_seam(
    tmp_path: Path, monkeypatch
) -> None:
    from scripts import analyze_family_errors

    output_dir = tmp_path / "analysis"
    checkpoint_path = tmp_path / "family.ckpt"
    export_path = _write_export(
        tmp_path / "canonical-export.json",
        checkpoint_path=str(checkpoint_path),
    )
    captured: dict[str, object] = {}

    def _fake_evaluate_checkpoints_with_eval_config(
        *,
        checkpoint_paths: dict[str, str],
        eval_cfg: object,
        prediction_output_dir: str | Path,
    ) -> dict[str, Path]:
        captured["checkpoint_paths"] = checkpoint_paths
        captured["prediction_output_dir"] = Path(prediction_output_dir)
        captured["eval_output_dir"] = str(eval_cfg.paths.output_dir)
        captured["eval_log_dir"] = str(eval_cfg.paths.log_dir)
        return {"family_analysis": export_path}

    monkeypatch.setattr(
        analyze_family_errors,
        "evaluate_checkpoints_with_eval_config",
        _fake_evaluate_checkpoints_with_eval_config,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scripts/analyze_family_errors.py",
            "--model",
            f"FamilyEval=gnn:{checkpoint_path}",
            "--output-dir",
            str(output_dir),
        ],
    )
    analyze_family_errors.main()

    assert captured["checkpoint_paths"] == {"family_analysis": str(checkpoint_path)}
    assert captured["prediction_output_dir"] == output_dir / "predictions"
    assert captured["eval_output_dir"] == str(output_dir)
    assert captured["eval_log_dir"] == str(output_dir)
    assert any(output_dir.glob("*-family-report.md"))
