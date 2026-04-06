"""Tests for the user-facing training launcher."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def write_sample_pfasbench_dataset(root: Path) -> Path:
    """Create a tiny PFASBench-style dataset for train launcher smoke tests."""
    raw_dir = root / "pfasbench" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pfasbench.csv").write_text(
        "\n".join(
            [
                "smiles,name,logS,logP,pKa",
                "C(=O)(C(F)(F)F)O,TFA,-0.5,0.5,0.5",
                "C(=O)(C(C(F)(F)F)(F)F)O,PFPA,-1.0,1.0,0.6",
                "C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O,PFBA,-1.5,1.5,0.7",
                "C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O,PFPeA,-2.0,2.0,0.8",
                "c1ccccc1,Benzene,-0.8,2.1,15.0",
                "c1ccc(cc1)O,Phenol,-0.5,1.5,9.9",
            ]
        )
        + "\n"
    )
    return root


def test_train_script_uses_canonical_config_and_persists_resolved_hydra_config(
    tmp_path: Path,
) -> None:
    """The user-facing train script should compose config groups and save the resolved Hydra config."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "scripts/train.py",
            "model=painn",
            "data=pfasbench",
            "train.run_test=false",
            "+trainer.fast_dev_run=true",
            "trainer.accelerator=cpu",
            "trainer.devices=1",
            "data.batch_size=2",
            "data.num_workers=0",
            "data.root=" + str(dataset_root),
            "data.split=random",
            "data.train_frac=0.5",
            "data.val_frac=0.25",
            "data.test_frac=0.25",
            "extras.print_config=false",
            "extras.enforce_tags=false",
            "hydra.run.dir=" + str(run_dir),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[2],
    )

    hydra_config = run_dir / ".hydra" / "config.yaml"
    assert hydra_config.exists()

    resolved_cfg = yaml.safe_load(hydra_config.read_text())
    assert resolved_cfg["model"]["_target_"] == "gnn.models.MoMLCAModel"
    assert (
        resolved_cfg["model"]["backbone"]["_target_"] == "gnn.models.backbones.PaiNNStageBackbone"
    )
    assert resolved_cfg["data"]["_target_"] == "gnn.data.datamodules.PFASBenchDataModule"
    assert resolved_cfg["trainer"]["fast_dev_run"] is True
    assert resolved_cfg["trainer"]["max_epochs"] == 10
    assert resolved_cfg["train"]["run_test"] is False
