"""Tests for the MoMLCA LightningModule."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from torch import nn
from torch_geometric.data import Batch, Data

from gnn.data.datamodules import PFASBenchDataModule
from gnn.models import MoMLCAModel, PaiNNStageBackbone
from src.train import train


def make_batch() -> Batch:
    """Create a small batched graph regression example."""
    graph_one = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0], [1.0]], dtype=torch.float32),
        y=torch.tensor([[1.0, float("nan"), 3.0]], dtype=torch.float32),
    )
    graph_two = Data(
        x=torch.tensor([[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]], dtype=torch.float32),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_attr=torch.tensor([[0.5], [0.5], [0.75], [0.75]], dtype=torch.float32),
        y=torch.tensor([[2.0, 4.0, float("nan")]], dtype=torch.float32),
    )
    return Batch.from_data_list([graph_one, graph_two])


def write_sample_pfasbench_dataset(root: Path) -> Path:
    """Create a tiny PFASBench-style dataset for trainer smoke tests."""
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


class RecordingBackbone(nn.Module):
    """Backbone test double that aggregates node features per graph."""

    def __init__(self) -> None:
        super().__init__()
        self.last_batch: Batch | None = None

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        self.last_batch = batch
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(batch.x[mask].mean(dim=0))
        return {
            "graph_features": torch.stack(graph_features, dim=0),
        }


class RecordingHead(nn.Module):
    """Head test double that records the features it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.last_inputs: torch.Tensor | None = None
        self.proj = nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [0.0, 1.0],
                        [1.0, 1.0],
                    ],
                    dtype=torch.float32,
                )
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.last_inputs = features
        return self.proj(features)


class StaticHead(nn.Module):
    """Head that returns fixed predictions for loss/metric assertions."""

    def __init__(self, predictions: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("predictions", predictions)
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        return self.predictions + (self.anchor * 0.0)


def test_forward_routes_backbone_outputs_to_named_heads() -> None:
    """The model should call the backbone first, then each named head."""
    batch = make_batch()
    backbone = RecordingBackbone()
    property_head = RecordingHead()
    model = MoMLCAModel(
        backbone=backbone,
        heads={"property": property_head},
        property_names=["logS", "logP", "pKa"],
    )

    outputs = model.forward(batch)

    assert backbone.last_batch is batch
    assert set(outputs) == {"backbone", "predictions"}
    assert set(outputs["predictions"]) == {"property"}
    assert property_head.last_inputs is not None
    assert torch.equal(property_head.last_inputs, outputs["backbone"]["graph_features"])
    assert outputs["predictions"]["property"].shape == (batch.num_graphs, 3)


def test_step_methods_mask_nan_targets_and_log_regression_metrics() -> None:
    """Training, validation, and test steps should log masked regression metrics."""
    batch = make_batch()
    predictions = torch.tensor(
        [
            [1.5, 1.0, 1.0],
            [1.0, 5.0, 7.0],
        ],
        dtype=torch.float32,
    )
    logged: list[str] = []
    model = MoMLCAModel(
        backbone=RecordingBackbone(),
        heads={"property": StaticHead(predictions)},
        property_names=["logS", "logP", "pKa"],
    )
    model.log = lambda name, value, **_: logged.append(name)  # type: ignore[method-assign]

    train_loss = model.training_step(batch, batch_idx=0)
    model.validation_step(batch, batch_idx=0)
    model.test_step(batch, batch_idx=0)

    expected_loss = torch.tensor((0.25 + 4.0 + 1.0 + 1.0) / 4.0, dtype=torch.float32)
    expected_mae = torch.tensor((0.5 + 2.0 + 1.0 + 1.0) / 4.0, dtype=torch.float32)

    assert torch.isclose(train_loss.detach(), expected_loss)
    assert torch.isclose(model._compute_masked_mae(predictions, batch.y).detach(), expected_mae)
    assert set(logged) == {
        "train/loss",
        "train/mae",
        "val/loss",
        "val/mae",
        "test/loss",
        "test/mae",
    }


def test_configure_optimizers_defaults_to_adamw_and_plateau_monitor() -> None:
    """The optimizer wiring should default to AdamW and expose val/loss for plateau schedulers."""
    model = MoMLCAModel(
        backbone=RecordingBackbone(),
        heads={"property": RecordingHead()},
        learning_rate=0.05,
        weight_decay=0.1,
        scheduler=partial(torch.optim.lr_scheduler.ReduceLROnPlateau, mode="min"),
    )

    optimizer_config = model.configure_optimizers()

    optimizer = optimizer_config["optimizer"]
    scheduler = optimizer_config["lr_scheduler"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 0.05
    assert optimizer.defaults["weight_decay"] == 0.1
    assert scheduler["monitor"] == "val/loss"
    assert isinstance(scheduler["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_hydra_instantiates_model_with_pfasbench_config() -> None:
    """Hydra should instantiate the GNN model config alongside PFASBench data config."""
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=momlca", "data=pfasbench"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

        model = hydra.utils.instantiate(cfg.model)
        datamodule = hydra.utils.instantiate(cfg.data)

    GlobalHydra.instance().clear()

    assert isinstance(model, MoMLCAModel)
    assert isinstance(datamodule, PFASBenchDataModule)


def test_hydra_instantiates_distinct_painn_stage_backbone() -> None:
    """The `model=painn` override should select a concrete backbone config."""
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=painn", "data=pfasbench"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

        model = hydra.utils.instantiate(cfg.model)

    GlobalHydra.instance().clear()

    assert isinstance(model, MoMLCAModel)
    assert isinstance(model.backbone, PaiNNStageBackbone)


def test_train_fast_dev_run_with_momlca_uses_default_callbacks(tmp_path: Path) -> None:
    """The real training path should work with `model=momlca` and the default callbacks."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=momlca", "data=pfasbench"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.paths.output_dir = str(tmp_path / "outputs")
            cfg.paths.log_dir = str(tmp_path / "outputs")
            cfg.data.root = str(dataset_root)
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.split = "random"
            cfg.data.train_frac = 0.5
            cfg.data.val_frac = 0.25
            cfg.data.test_frac = 0.25
            cfg.trainer.fast_dev_run = True
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.train.run_test = False
            cfg.logger = None
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False

        HydraConfig().set_config(cfg)
        metric_dict, _ = train(cfg)

    GlobalHydra.instance().clear()

    assert "train/loss" in metric_dict
    assert "val/loss" in metric_dict
