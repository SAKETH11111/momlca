import hydra
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig


def test_train_config(cfg_train: DictConfig) -> None:
    """Tests the training configuration provided by the `cfg_train` pytest fixture.

    :param cfg_train: A DictConfig containing a valid training configuration.
    """
    assert cfg_train
    assert cfg_train.data
    assert cfg_train.model
    assert cfg_train.trainer

    HydraConfig().set_config(cfg_train)

    hydra.utils.instantiate(cfg_train.data)
    hydra.utils.instantiate(cfg_train.model)
    hydra.utils.instantiate(cfg_train.trainer)


def test_eval_config(cfg_eval: DictConfig) -> None:
    """Tests the evaluation configuration provided by the `cfg_eval` pytest fixture.

    :param cfg_train: A DictConfig containing a valid evaluation configuration.
    """
    assert cfg_eval
    assert cfg_eval.data
    assert cfg_eval.model
    assert cfg_eval.trainer

    HydraConfig().set_config(cfg_eval)

    hydra.utils.instantiate(cfg_eval.data)
    hydra.utils.instantiate(cfg_eval.model)
    hydra.utils.instantiate(cfg_eval.trainer)


def test_canonical_train_entrypoint_supports_gnn_overrides() -> None:
    """The canonical train entrypoint should compose the new runtime train group."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=painn", "data=pfasbench", "trainer.max_epochs=5"],
        )
        cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

    GlobalHydra.instance().clear()

    assert cfg.train.run_train is True
    assert cfg.train.run_test is True
    assert cfg.model._target_ == "gnn.models.MoMLCAModel"
    assert cfg.data._target_ == "gnn.data.datamodules.PFASBenchDataModule"
    assert cfg.trainer.max_epochs == 5


def test_canonical_train_entrypoint_defaults_to_project_training_stack() -> None:
    """The canonical train config should default to the repo's GNN/PFASBench stack."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(config_name="config.yaml", return_hydra_config=True, overrides=[])

    GlobalHydra.instance().clear()

    assert cfg.model._target_ == "gnn.models.MoMLCAModel"
    assert cfg.data._target_ == "gnn.data.datamodules.PFASBenchDataModule"


@pytest.mark.parametrize("logger_name", ["wandb", "tensorboard", "many_loggers"])
def test_logger_presets_compose_with_canonical_config(logger_name: str) -> None:
    """Logger groups should compose with the canonical Hydra entrypoint."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=[f"logger={logger_name}"],
        )

    GlobalHydra.instance().clear()

    assert cfg.logger is not None
    if logger_name in ("wandb", "many_loggers"):
        assert cfg.logger.wandb.project == "moml"
    if logger_name == "tensorboard":
        assert "tensorboard" in cfg.logger
