import os
from pathlib import Path

import pytest
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict

from src.train import train
from tests.helpers.pfasbench import write_sample_pfasbench_dataset
from tests.helpers.resume_probe import ResumeProbeCallback
from tests.helpers.run_if import RunIf


def test_train_fast_dev_run(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "cpu"
    train(cfg_train)


@RunIf(min_gpus=1)
def test_train_fast_dev_run_gpu(cfg_train: DictConfig) -> None:
    """Run for 1 train, val and test step on GPU.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.fast_dev_run = True
        cfg_train.trainer.accelerator = "gpu"
    train(cfg_train)


@RunIf(min_gpus=1)
@pytest.mark.slow
def test_train_epoch_gpu_amp(cfg_train: DictConfig) -> None:
    """Train 1 epoch on GPU with mixed-precision.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.accelerator = "gpu"
        cfg_train.trainer.precision = 16
    train(cfg_train)


@pytest.mark.slow
def test_train_epoch_double_val_loop(cfg_train: DictConfig) -> None:
    """Train 1 epoch with validation loop twice per epoch.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.val_check_interval = 0.5
    train(cfg_train)


@pytest.mark.slow
def test_train_ddp_sim(cfg_train: DictConfig) -> None:
    """Simulate DDP (Distributed Data Parallel) on 2 CPU processes.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    HydraConfig().set_config(cfg_train)
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 2
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 2
        cfg_train.trainer.strategy = "ddp_spawn"
    train(cfg_train)


@pytest.mark.slow
def test_train_resume(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Run 1 epoch, finish, and resume for another epoch.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    with open_dict(cfg_train):
        cfg_train.trainer.max_epochs = 1
        cfg_train.train.run_test = False
        cfg_train.test = False
        cfg_train.callbacks.resume_probe = {
            "_target_": "tests.helpers.resume_probe.ResumeProbeCallback"
        }

    HydraConfig().set_config(cfg_train)
    _, object_dict_1 = train(cfg_train)

    checkpoint_dir = tmp_path / "checkpoints"
    files = os.listdir(checkpoint_dir)
    assert "best.ckpt" in files
    assert "last.ckpt" in files
    assert "epoch_000.ckpt" in files

    primary_checkpoint_callback = object_dict_1["trainer"].checkpoint_callback
    assert Path(primary_checkpoint_callback.best_model_path).name == "best.ckpt"
    assert Path(primary_checkpoint_callback.best_model_path).exists()

    first_probe = next(
        callback
        for callback in object_dict_1["callbacks"]
        if isinstance(callback, ResumeProbeCallback)
    )
    assert first_probe.train_start_epoch == 0
    assert first_probe.train_start_global_step == 0
    first_global_step = object_dict_1["trainer"].global_step

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(checkpoint_dir / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    _, object_dict_2 = train(cfg_train)

    files = os.listdir(checkpoint_dir)
    assert "epoch_001.ckpt" in files
    assert "epoch_002.ckpt" not in files

    second_probe = next(
        callback
        for callback in object_dict_2["callbacks"]
        if isinstance(callback, ResumeProbeCallback)
    )
    assert second_probe.train_start_epoch == 1
    assert second_probe.train_start_global_step == first_global_step
    assert object_dict_2["trainer"].global_step > first_global_step


@pytest.mark.slow
def test_train_resume_on_pfasbench_stack(tmp_path: Path, cfg_train: DictConfig) -> None:
    """Resume training on the canonical PFASBench/MoMLCA stack."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")

    with open_dict(cfg_train):
        cfg_train.model = OmegaConf.load(project_root / "configs/model/painn.yaml")
        cfg_train.data = OmegaConf.load(project_root / "configs/data/pfasbench.yaml")
        cfg_train.paths.output_dir = str(tmp_path)
        cfg_train.paths.log_dir = str(tmp_path)
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.data.root = str(dataset_root)
        cfg_train.data.batch_size = 2
        cfg_train.data.num_workers = 0
        cfg_train.data.split = "random"
        cfg_train.data.train_frac = 0.5
        cfg_train.data.val_frac = 0.25
        cfg_train.data.test_frac = 0.25
        cfg_train.train.run_test = False
        cfg_train.test = False
        cfg_train.callbacks.resume_probe = {
            "_target_": "tests.helpers.resume_probe.ResumeProbeCallback"
        }

    HydraConfig().set_config(cfg_train)
    _, object_dict_1 = train(cfg_train)

    checkpoint_dir = tmp_path / "checkpoints"
    assert (checkpoint_dir / "best.ckpt").exists()
    assert (checkpoint_dir / "last.ckpt").exists()
    assert (checkpoint_dir / "epoch_000.ckpt").exists()

    first_probe = next(
        callback
        for callback in object_dict_1["callbacks"]
        if isinstance(callback, ResumeProbeCallback)
    )
    assert first_probe.train_start_epoch == 0
    assert first_probe.train_start_global_step == 0
    first_global_step = object_dict_1["trainer"].global_step

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(checkpoint_dir / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    _, object_dict_2 = train(cfg_train)

    second_probe = next(
        callback
        for callback in object_dict_2["callbacks"]
        if isinstance(callback, ResumeProbeCallback)
    )
    assert second_probe.train_start_epoch == 1
    assert second_probe.train_start_global_step == first_global_step
    assert object_dict_2["trainer"].global_step > first_global_step


def test_train_accepts_finetune_backbone_config_without_using_ckpt_path(
    tmp_path: Path, cfg_train: DictConfig
) -> None:
    """Canonical train() should honor pretrained-backbone init separately from resume ckpt_path."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    pretrained_path = tmp_path / "pretrained-backbone.pt"

    torch.save(
        {
            "backbone.linear.weight": torch.full((4, 22), 1.5),
            "backbone.linear.bias": torch.full((4,), 0.25),
        },
        pretrained_path,
    )

    with open_dict(cfg_train):
        cfg_train.model = OmegaConf.load(project_root / "configs/model/momlca.yaml")
        cfg_train.data = OmegaConf.load(project_root / "configs/data/pfasbench.yaml")
        cfg_train.paths.output_dir = str(tmp_path)
        cfg_train.paths.log_dir = str(tmp_path)
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.data.root = str(dataset_root)
        cfg_train.data.batch_size = 2
        cfg_train.data.num_workers = 0
        cfg_train.data.split = "random"
        cfg_train.data.train_frac = 0.5
        cfg_train.data.val_frac = 0.25
        cfg_train.data.test_frac = 0.25
        cfg_train.train.run_test = False
        cfg_train.test = False
        cfg_train.ckpt_path = None
        cfg_train.model.backbone = {
            "_target_": "tests.helpers.transfer_learning.TinyBackbone",
            "input_dim": 22,
            "hidden_dim": 4,
        }
        cfg_train.model.learning_rate = 0.0001
        cfg_train.model.pretrained_backbone.checkpoint_path = str(pretrained_path)
        cfg_train.model.pretrained_backbone.checkpoint_format = "state_dict"
        cfg_train.model.pretrained_backbone.backbone_key_prefix = "backbone."
        cfg_train.model.pretrained_backbone.freeze_backbone = True

    HydraConfig().set_config(cfg_train)
    _, object_dict = train(cfg_train)

    model = object_dict["model"]
    assert model._pretrained_backbone_loaded is True
    assert object_dict["cfg"].ckpt_path is None
    assert torch.equal(model.backbone.linear.weight, torch.full((4, 22), 1.5))
    assert torch.equal(model.backbone.linear.bias, torch.full((4,), 0.25))
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())


def test_train_resume_does_not_require_original_pretrained_checkpoint(
    tmp_path: Path, cfg_train: DictConfig
) -> None:
    """Exact resume should not depend on the original pretrained-backbone artifact."""
    project_root = Path(__file__).resolve().parents[1]
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    pretrained_path = tmp_path / "pretrained-backbone.pt"

    torch.save(
        {
            "backbone.linear.weight": torch.full((4, 22), 0.5),
            "backbone.linear.bias": torch.full((4,), -0.5),
        },
        pretrained_path,
    )

    with open_dict(cfg_train):
        cfg_train.model = OmegaConf.load(project_root / "configs/model/momlca.yaml")
        cfg_train.data = OmegaConf.load(project_root / "configs/data/pfasbench.yaml")
        cfg_train.paths.output_dir = str(tmp_path)
        cfg_train.paths.log_dir = str(tmp_path)
        cfg_train.trainer.max_epochs = 1
        cfg_train.trainer.limit_train_batches = 1
        cfg_train.trainer.limit_val_batches = 1
        cfg_train.trainer.num_sanity_val_steps = 0
        cfg_train.trainer.accelerator = "cpu"
        cfg_train.trainer.devices = 1
        cfg_train.data.root = str(dataset_root)
        cfg_train.data.batch_size = 2
        cfg_train.data.num_workers = 0
        cfg_train.data.split = "random"
        cfg_train.data.train_frac = 0.5
        cfg_train.data.val_frac = 0.25
        cfg_train.data.test_frac = 0.25
        cfg_train.train.run_test = False
        cfg_train.test = False
        cfg_train.ckpt_path = None
        cfg_train.callbacks.resume_probe = {
            "_target_": "tests.helpers.resume_probe.ResumeProbeCallback"
        }
        cfg_train.model.backbone = {
            "_target_": "tests.helpers.transfer_learning.TinyBackbone",
            "input_dim": 22,
            "hidden_dim": 4,
        }
        cfg_train.model.pretrained_backbone.checkpoint_path = str(pretrained_path)
        cfg_train.model.pretrained_backbone.checkpoint_format = "state_dict"
        cfg_train.model.pretrained_backbone.backbone_key_prefix = "backbone."
        cfg_train.model.pretrained_backbone.freeze_backbone = False

    HydraConfig().set_config(cfg_train)
    _, object_dict_1 = train(cfg_train)

    checkpoint_dir = tmp_path / "checkpoints"
    first_probe = next(
        callback
        for callback in object_dict_1["callbacks"]
        if isinstance(callback, ResumeProbeCallback)
    )
    assert first_probe.train_start_epoch == 0
    assert object_dict_1["model"]._pretrained_backbone_loaded is True
    first_global_step = object_dict_1["trainer"].global_step

    pretrained_path.unlink()

    with open_dict(cfg_train):
        cfg_train.ckpt_path = str(checkpoint_dir / "last.ckpt")
        cfg_train.trainer.max_epochs = 2

    _, object_dict_2 = train(cfg_train)

    second_probe = next(
        callback
        for callback in object_dict_2["callbacks"]
        if isinstance(callback, ResumeProbeCallback)
    )
    assert second_probe.train_start_epoch == 1
    assert second_probe.train_start_global_step == first_global_step
    assert object_dict_2["model"]._pretrained_backbone_loaded is False
