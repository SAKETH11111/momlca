import json
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import (
    _disable_pretrained_backbone_for_resume,
    _supports_prediction_collection,
    evaluate,
)
from src.train import train


def write_sample_pfasbench_dataset(root: Path) -> Path:
    """Create a tiny PFASBench-style dataset for eval smoke tests."""
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
                "C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHxA,-2.5,2.5,0.9",
                "C(=O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHpA,-3.0,3.0,1.0",
                "c1ccccc1,Benzene,-0.8,2.1,15.0",
                "c1ccc(cc1)O,Phenol,-0.5,1.5,9.9",
            ]
        )
        + "\n"
    )
    return root


def test_eval_checkpoint_path_disables_pretrained_backbone_checkpoint() -> None:
    """Exact-checkpoint evaluation should ignore transfer-learning checkpoint config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=["model=momlca", "data=pfasbench_scaffold", "ckpt_path=/tmp/best.ckpt"],
        )

    with open_dict(cfg):
        cfg.model.pretrained_backbone.checkpoint_path = "/tmp/pretrained.ckpt"

    skipped_checkpoint = _disable_pretrained_backbone_for_resume(cfg)

    assert skipped_checkpoint == "/tmp/pretrained.ckpt"
    assert cfg.model.pretrained_backbone.checkpoint_path is None
    GlobalHydra.instance().clear()


def test_supports_prediction_collection_blocks_spawn_and_fork_strategies() -> None:
    """Prediction collection should be disabled for spawn/fork launchers."""
    spawn_trainer = SimpleNamespace(
        strategy=SimpleNamespace(strategy_name="ddp_spawn", launcher=SimpleNamespace())
    )
    fork_launcher = type("ForkLauncher", (), {})()
    fork_trainer = SimpleNamespace(
        strategy=SimpleNamespace(strategy_name="ddp", launcher=fork_launcher)
    )
    ddp_trainer = SimpleNamespace(strategy=SimpleNamespace(strategy_name="ddp", launcher=None))

    assert _supports_prediction_collection(spawn_trainer) is False
    assert _supports_prediction_collection(fork_trainer) is False
    assert _supports_prediction_collection(ddp_trainer) is True


@pytest.mark.slow
def test_train_eval(tmp_path: Path, cfg_train: DictConfig, cfg_eval: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_train.paths.output_dir == cfg_eval.paths.output_dir

    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.test = True

    HydraConfig().set_config(cfg_train)
    train_metric_dict, object_dict = train(cfg_train)

    assert "last.ckpt" in os.listdir(tmp_path / "checkpoints")
    best_checkpoint = Path(object_dict["trainer"].checkpoint_callback.best_model_path)
    assert best_checkpoint.name == "best.ckpt"
    assert best_checkpoint.exists()

    with open_dict(cfg_eval):
        cfg_eval.data = cfg_train.data
        cfg_eval.model = cfg_train.model
        cfg_eval.ckpt_path = str(best_checkpoint)

    HydraConfig().set_config(cfg_eval)
    test_metric_dict, _ = evaluate(cfg_eval)

    assert test_metric_dict["test/acc"] > 0.0
    assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001


@pytest.mark.slow
def test_eval_pfasbench_exports_test_predictions_and_metric_contract(tmp_path: Path) -> None:
    """PFASBench eval should preserve metric keys and export test-only JSON predictions."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    train_output_dir = tmp_path / "train-run"
    eval_output_dir = tmp_path / "eval-run"

    with initialize(version_base="1.3", config_path="../configs"):
        cfg_train = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=momlca", "data=pfasbench_scaffold"],
        )
    with open_dict(cfg_train):
        cfg_train.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
        cfg_train.paths.output_dir = str(train_output_dir)
        cfg_train.paths.log_dir = str(train_output_dir)
        cfg_train.data.root = str(dataset_root)
        cfg_train.data.batch_size = 2
        cfg_train.data.num_workers = 0
        cfg_train.data.split = "random"
        cfg_train.data.train_frac = 0.5
        cfg_train.data.val_frac = 0.25
        cfg_train.data.test_frac = 0.25
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.train.run_test = False
        cfg_train.logger = None
        cfg_train.extras.print_config = False
        cfg_train.extras.enforce_tags = False

    HydraConfig().set_config(cfg_train)
    train(cfg_train)
    checkpoint_path = train_output_dir / "checkpoints" / "last.ckpt"
    assert checkpoint_path.exists()

    with initialize(version_base="1.3", config_path="../configs"):
        cfg_eval = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "model=momlca",
                "data=pfasbench_scaffold",
                f"ckpt_path={checkpoint_path}",
            ],
        )
    with open_dict(cfg_eval):
        cfg_eval.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
        cfg_eval.paths.output_dir = str(eval_output_dir)
        cfg_eval.paths.log_dir = str(eval_output_dir)
        cfg_eval.data.root = str(dataset_root)
        cfg_eval.data.batch_size = 2
        cfg_eval.data.num_workers = 0
        cfg_eval.data.split = "random"
        cfg_eval.data.seed = 42
        cfg_eval.data.train_frac = 0.5
        cfg_eval.data.val_frac = 0.25
        cfg_eval.data.test_frac = 0.25
        cfg_eval.trainer.accelerator = "cpu"
        cfg_eval.trainer.devices = 1
        cfg_eval.trainer.limit_test_batches = 1.0
        cfg_eval.logger = None
        cfg_eval.extras.print_config = False
        cfg_eval.extras.enforce_tags = False
        cfg_eval.export_predictions = True
        cfg_eval.log_prediction_artifact = False

    HydraConfig().set_config(cfg_eval)
    metric_dict, _ = evaluate(cfg_eval)

    for metric_name in (
        "test/mae_mean",
        "test/rmse_mean",
        "test/r2_mean",
        "test/pearson_mean",
        "test/spearman_mean",
    ):
        assert metric_name in metric_dict

    export_files = sorted((eval_output_dir / "predictions").glob("*.json"))
    assert len(export_files) == 1
    export_payload = json.loads(export_files[0].read_text())

    datamodule = hydra.utils.instantiate(cfg_eval.data)
    datamodule.setup(stage="test")
    assert datamodule.test_dataset is not None

    assert export_payload["metadata"]["split"] == "test"
    assert export_payload["metadata"]["checkpoint_path"] == str(checkpoint_path)
    assert export_payload["metadata"]["num_records"] == len(datamodule.test_dataset)
    assert export_payload["metadata"]["property_names"] == ["logS", "logP", "pKa"]
    assert len(export_payload["records"]) == len(datamodule.test_dataset)

    for record in export_payload["records"]:
        assert record["split"] == "test"
        assert record["checkpoint_path"] == str(checkpoint_path)
        assert set(record["targets"]) == {"logS", "logP", "pKa"}
        assert set(record["predictions"]) == {"logS", "logP", "pKa"}
        assert "smiles" in record
        assert "name" in record
        assert "inchikey" in record

    GlobalHydra.instance().clear()
