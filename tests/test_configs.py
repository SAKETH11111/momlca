from pathlib import Path

import hydra
import pytest
import rootutils
import yaml
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tests.helpers.pretrained_artifacts import TRACKED_PAINN_STAGE_ARTIFACT_RELATIVE_PATH


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
    assert cfg.model.backbone._target_ == "gnn.models.backbones.PaiNNBackbone"
    assert cfg.data._target_ == "gnn.data.datamodules.PFASBenchDataModule"
    assert cfg.trainer.max_epochs == 5


def test_canonical_train_entrypoint_supports_gin_override() -> None:
    """The canonical train entrypoint should compose the new GIN model preset."""
    try:
        with initialize(version_base="1.3", config_path="../configs"):
            cfg = compose(
                config_name="config.yaml",
                return_hydra_config=True,
                overrides=["model=gin", "data=pfasbench"],
            )
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
    finally:
        GlobalHydra.instance().clear()

    assert cfg.model._target_ == "gnn.models.MoMLCAModel"
    assert cfg.model.backbone._target_ == "gnn.models.backbones.GINBackbone"
    assert cfg.model.backbone.input_dim == 22
    assert cfg.model.backbone.hidden_channels == 128
    assert cfg.model.backbone.num_layers == 4
    assert cfg.data._target_ == "gnn.data.datamodules.PFASBenchDataModule"


@pytest.mark.parametrize(
    ("model_name", "backbone_target"),
    [
        ("gin", "gnn.models.backbones.GINBackbone"),
        ("painn", "gnn.models.backbones.PaiNNBackbone"),
    ],
)
def test_backbone_presets_auto_match_property_head_input_dim(
    model_name: str, backbone_target: str
) -> None:
    """Backbone presets should keep default property-head width aligned automatically."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=[f"model={model_name}", "data=pfasbench"],
        )
        cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
        model = hydra.utils.instantiate(cfg.model)

    GlobalHydra.instance().clear()

    assert cfg.model.backbone._target_ == backbone_target
    assert model.heads["property"].input_dim == model.backbone.output_dim


def test_multiseed_train_preset_composes_with_canonical_entrypoint() -> None:
    """The optional multiseed train preset should compose without changing the entrypoint."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["train=multiseed"],
        )

    GlobalHydra.instance().clear()

    assert cfg.train.multiseed.enabled is True
    assert cfg.train.multiseed.metrics_filename == "multiseed_metrics.json"
    assert "multiseed" in cfg.train.tags


@pytest.mark.parametrize(
    ("experiment_name", "expected_backbone_target"),
    [
        ("pfasbench_finetune", "gnn.models.backbones.PaiNNStageBackbone"),
        ("pfasbench_finetune_momlca", "gnn.models.backbones.PaiNNStageBackbone"),
    ],
)
def test_finetune_experiments_compose_with_lower_lr_and_pretrained_backbone(
    experiment_name: str, expected_backbone_target: str | None
) -> None:
    """Fine-tune experiments should stay on the canonical entrypoint and expose transfer config."""
    with initialize(version_base="1.3", config_path="../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=[f"experiment={experiment_name}"],
        )

    GlobalHydra.instance().clear()

    assert cfg.model._target_ == "gnn.models.MoMLCAModel"
    assert cfg.model.learning_rate == 0.0001
    assert (
        cfg.model.pretrained_backbone.checkpoint_path == TRACKED_PAINN_STAGE_ARTIFACT_RELATIVE_PATH
    )
    assert cfg.model.pretrained_backbone.checkpoint_format == "state_dict"
    assert cfg.model.pretrained_backbone.backbone_key_prefix == "backbone."
    assert cfg.ckpt_path is None
    assert cfg.model.backbone._target_ == expected_backbone_target
    assert cfg.model.backbone.use_positions is False


def test_tracked_pretrained_artifact_metadata_pins_repo_conversion_schema() -> None:
    """Tracked artifact metadata should pin the repo-side conversion schema exactly."""
    metadata_path = (
        Path(__file__).resolve().parents[1]
        / "artifacts/pretrained/isdpainn_random_split_painn_stage_backbone.metadata.yaml"
    )

    metadata = yaml.safe_load(metadata_path.read_text())
    conversion = metadata["conversion"]

    assert conversion["schema_version"] == 1
    assert conversion["repo_head_commit"]
    assert conversion["script_sha256"]
    assert conversion["constants_module"] == "src/gnn/data/transforms/constants.py"
    assert conversion["constants_module_sha256"]
    assert conversion["schema_sha256"]
    assert conversion["repo_feature_schema"]["atom_feature_dim"] == 22
    assert conversion["repo_feature_schema"]["atom_atomic_number_slice"] == [0, 10]
    assert conversion["repo_feature_schema"]["target_hidden_channels"] == 128
    assert metadata["source"]["upstream_snapshot"]["checkpoint_retrieved_at"]
    assert metadata["source"]["upstream_snapshot"]["config_retrieved_at"]


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
