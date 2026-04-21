"""Tests for the MoMLCA LightningModule."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import hydra
import pytest
import rootutils
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict
from torch import nn
from torch_geometric.data import Batch, Data

from gnn.data.datamodules import PFASBenchDataModule
from gnn.models import GINBackbone, MoMLCAModel, PaiNNStageBackbone
from gnn.models.backbones import BaseBackbone
from src.train import train
from tests.helpers.transfer_learning import TinyBackbone


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


class RecordingBackbone(BaseBackbone):
    """Backbone test double that aggregates node features per graph."""

    def __init__(self) -> None:
        super().__init__()
        self.last_batch: Batch | None = None

    @property
    def output_dim(self) -> int:
        return 2

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        self.last_batch = batch
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(batch.x[mask].mean(dim=0))
        return {
            "node_features": batch.x,
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


class _PlainModuleBackbone(nn.Module):
    """Invalid backbone that does not implement the shared BaseBackbone contract."""

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(batch.x[mask].mean(dim=0))
        return {
            "node_features": batch.x,
            "graph_features": torch.stack(graph_features, dim=0),
        }


class _MissingNodeFeaturesBackbone(BaseBackbone):
    """Backbone that omits node features to exercise output-contract validation."""

    @property
    def output_dim(self) -> int:
        return 2

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(batch.x[mask].mean(dim=0))
        return {"graph_features": torch.stack(graph_features, dim=0)}


class _TensorOnlyBackbone(BaseBackbone):
    """Backbone that returns a bare tensor instead of the required mapping."""

    @property
    def output_dim(self) -> int:
        return 2

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(batch.x[mask].mean(dim=0))
        return torch.stack(graph_features, dim=0)  # type: ignore[return-value]


class _MismatchedOutputDimBackbone(BaseBackbone):
    """Backbone with mismatched output_dim for graph features."""

    @property
    def output_dim(self) -> int:
        return 3

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(batch.x[mask].mean(dim=0))
        return {
            "node_features": batch.x,
            "graph_features": torch.stack(graph_features, dim=0),
        }


def save_lightning_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> Path:
    """Persist a minimal Lightning-style checkpoint for transfer-loading tests."""
    torch.save({"state_dict": state_dict, "epoch": 0, "global_step": 0}, path)
    return path


def save_state_dict_checkpoint(path: Path, state_dict: dict[str, torch.Tensor]) -> Path:
    """Persist a plain state dict checkpoint for transfer-loading tests."""
    torch.save(state_dict, path)
    return path


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

    model.on_train_epoch_start()
    train_loss = model.training_step(batch, batch_idx=0)
    model.on_train_epoch_end()
    model.on_validation_epoch_start()
    model.validation_step(batch, batch_idx=0)
    model.on_validation_epoch_end()
    model.on_test_epoch_start()
    model.test_step(batch, batch_idx=0)
    model.on_test_epoch_end()

    expected_loss = torch.tensor((0.25 + 4.0 + 1.0 + 1.0) / 4.0, dtype=torch.float32)
    expected_mae = torch.tensor((0.5 + 2.0 + 1.0 + 1.0) / 4.0, dtype=torch.float32)

    assert torch.isclose(train_loss.detach(), expected_loss)
    assert torch.isclose(model._compute_masked_mae(predictions, batch.y).detach(), expected_mae)
    assert set(logged) == {
        "train/loss",
        "train/mae",
        "train/mae_logP",
        "train/mae_logS",
        "train/mae_pKa",
        "train/rmse_logP",
        "train/rmse_logS",
        "train/rmse_pKa",
        "val/loss",
        "val/mae",
        "val/mae_logP",
        "val/mae_logS",
        "val/mae_pKa",
        "val/rmse_logP",
        "val/rmse_logS",
        "val/rmse_pKa",
        "test/loss",
        "test/mae",
        "test/mae_logP",
        "test/mae_logS",
        "test/mae_pKa",
        "test/rmse_logP",
        "test/rmse_logS",
        "test/rmse_pKa",
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


def test_model_rejects_backbones_that_do_not_implement_base_contract() -> None:
    """Arbitrary nn.Module backbones should not bypass the shared backbone interface."""
    with pytest.raises(TypeError, match="BaseBackbone"):
        MoMLCAModel(backbone=_PlainModuleBackbone())


def test_model_rejects_backbone_outputs_missing_node_features() -> None:
    """Backbones must provide both node and graph features under the shared schema."""
    batch = make_batch()
    model = MoMLCAModel(
        backbone=_MissingNodeFeaturesBackbone(),
        heads={"property": RecordingHead()},
    )

    with pytest.raises(ValueError, match="node_features"):
        model.forward(batch)


def test_model_rejects_backbone_outputs_that_are_not_mappings() -> None:
    """Backbones must return a mapping, not a bare tensor."""
    batch = make_batch()
    model = MoMLCAModel(
        backbone=_TensorOnlyBackbone(),
        heads={"property": RecordingHead()},
    )

    with pytest.raises(ValueError, match="Backbone must return a mapping"):
        model.forward(batch)


def test_model_rejects_graph_feature_width_mismatch_with_output_dim() -> None:
    """Backbones must keep graph feature width aligned with output_dim."""
    batch = make_batch()
    model = MoMLCAModel(
        backbone=_MismatchedOutputDimBackbone(),
        heads={"property": RecordingHead()},
    )

    with pytest.raises(ValueError, match="backbone.output_dim"):
        model.forward(batch)


def test_pretrained_lightning_checkpoint_loads_backbone_only_and_keeps_heads_fresh(
    tmp_path: Path,
) -> None:
    """Transfer loading should restore only backbone weights from Lightning checkpoints."""
    torch.manual_seed(7)
    source_model = MoMLCAModel(
        backbone=TinyBackbone(input_dim=2, hidden_dim=4),
        heads={"property": nn.Linear(4, 3)},
        property_names=["logS", "logP", "pKa"],
    )
    with torch.no_grad():
        source_model.backbone.linear.weight.fill_(2.5)  # type: ignore[union-attr]
        source_model.backbone.linear.bias.fill_(0.75)  # type: ignore[union-attr]
        source_model.heads["property"].weight.fill_(9.0)
        source_model.heads["property"].bias.fill_(4.0)

    checkpoint_path = save_lightning_checkpoint(
        tmp_path / "pretrained-lightning.ckpt",
        source_model.state_dict(),
    )

    torch.manual_seed(99)
    reference_model = MoMLCAModel(
        backbone=TinyBackbone(input_dim=2, hidden_dim=4),
        heads={"property": nn.Linear(4, 3)},
        property_names=["logS", "logP", "pKa"],
    )
    baseline_head_weight = reference_model.heads["property"].weight.detach().clone()
    baseline_head_bias = reference_model.heads["property"].bias.detach().clone()

    torch.manual_seed(99)
    model = MoMLCAModel(
        backbone=TinyBackbone(input_dim=2, hidden_dim=4),
        heads={"property": nn.Linear(4, 3)},
        property_names=["logS", "logP", "pKa"],
        pretrained_backbone={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "lightning",
            "backbone_key_prefix": "backbone.",
            "freeze_backbone": False,
        },
    )

    assert model._pretrained_backbone_loaded is True
    assert torch.equal(model.backbone.linear.weight, source_model.backbone.linear.weight)  # type: ignore[union-attr]
    assert torch.equal(model.backbone.linear.bias, source_model.backbone.linear.bias)  # type: ignore[union-attr]
    assert torch.equal(model.heads["property"].weight, baseline_head_weight)
    assert torch.equal(model.heads["property"].bias, baseline_head_bias)
    assert not torch.equal(model.heads["property"].weight, source_model.heads["property"].weight)


def test_pretrained_state_dict_checkpoint_supports_prefix_filtering(tmp_path: Path) -> None:
    """Plain state dict checkpoints should load when configured with a backbone prefix."""
    backbone = TinyBackbone(input_dim=2, hidden_dim=4)
    with torch.no_grad():
        backbone.linear.weight.fill_(1.25)
        backbone.linear.bias.fill_(-0.5)

    checkpoint_path = save_state_dict_checkpoint(
        tmp_path / "pretrained-state-dict.pt",
        {
            "backbone.linear.weight": backbone.linear.weight.detach().clone(),
            "backbone.linear.bias": backbone.linear.bias.detach().clone(),
            "heads.property.weight": torch.full((3, 4), 8.0),
        },
    )

    model = MoMLCAModel(
        backbone=TinyBackbone(input_dim=2, hidden_dim=4),
        heads={"property": nn.Linear(4, 3)},
        property_names=["logS", "logP", "pKa"],
        pretrained_backbone={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "state_dict",
            "backbone_key_prefix": "backbone.",
            "freeze_backbone": False,
        },
    )

    assert torch.equal(model.backbone.linear.weight, backbone.linear.weight)  # type: ignore[union-attr]
    assert torch.equal(model.backbone.linear.bias, backbone.linear.bias)  # type: ignore[union-attr]


def test_pretrained_real_lightning_checkpoint_from_train_stack_loads_backbone(
    tmp_path: Path,
) -> None:
    """Trainer-produced Lightning checkpoints should be reusable for transfer loading."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")
    run_dir = tmp_path / "source-run"

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=momlca", "data=pfasbench"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
            cfg.paths.output_dir = str(run_dir)
            cfg.paths.log_dir = str(run_dir)
            cfg.data.root = str(dataset_root)
            cfg.data.batch_size = 2
            cfg.data.num_workers = 0
            cfg.data.split = "random"
            cfg.data.train_frac = 0.5
            cfg.data.val_frac = 0.25
            cfg.data.test_frac = 0.25
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 1
            cfg.trainer.limit_val_batches = 1
            cfg.trainer.num_sanity_val_steps = 0
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.train.run_test = False
            cfg.logger = None
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.model.backbone = {
                "_target_": "tests.helpers.transfer_learning.TinyBackbone",
                "input_dim": 22,
                "hidden_dim": 4,
            }

        HydraConfig().set_config(cfg)
        train(cfg)

    checkpoint_path = run_dir / "checkpoints" / "last.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    GlobalHydra.instance().clear()

    model = MoMLCAModel(
        backbone=TinyBackbone(input_dim=22, hidden_dim=4),
        pretrained_backbone={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "lightning",
            "backbone_key_prefix": "backbone.",
            "freeze_backbone": False,
        },
    )

    assert model._pretrained_backbone_loaded is True
    assert torch.equal(
        model.backbone.linear.weight,  # type: ignore[union-attr]
        checkpoint["state_dict"]["backbone.linear.weight"],
    )
    assert torch.equal(
        model.backbone.linear.bias,  # type: ignore[union-attr]
        checkpoint["state_dict"]["backbone.linear.bias"],
    )


def test_pretrained_backbone_requires_usable_backbone_keys(tmp_path: Path) -> None:
    """Transfer loading should fail loudly when no backbone keys survive filtering."""
    checkpoint_path = save_state_dict_checkpoint(
        tmp_path / "invalid-state-dict.pt",
        {"heads.property.weight": torch.ones(3, 4)},
    )

    with pytest.raises(
        ValueError,
        match="Checkpoint did not contain any backbone parameters",
    ):
        MoMLCAModel(
            backbone=TinyBackbone(input_dim=2, hidden_dim=4),
            heads={"property": nn.Linear(4, 3)},
            property_names=["logS", "logP", "pKa"],
            pretrained_backbone={
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_format": "state_dict",
                "backbone_key_prefix": "backbone.",
                "freeze_backbone": False,
            },
        )


def test_freeze_backbone_keeps_backbone_fixed_and_head_trainable(tmp_path: Path) -> None:
    """Frozen backbones should stay fixed while the task head remains trainable."""
    checkpoint_path = save_state_dict_checkpoint(
        tmp_path / "frozen-backbone.pt",
        {
            "backbone.linear.weight": torch.full((4, 2), 0.75),
            "backbone.linear.bias": torch.full((4,), -0.25),
        },
    )
    model = MoMLCAModel(
        backbone=TinyBackbone(input_dim=2, hidden_dim=4),
        heads={"property": nn.Linear(4, 3)},
        property_names=["logS", "logP", "pKa"],
        pretrained_backbone={
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_format": "state_dict",
            "backbone_key_prefix": "backbone.",
            "freeze_backbone": True,
        },
    )
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    batch = make_batch()

    backbone_weight_before = model.backbone.linear.weight.detach().clone()  # type: ignore[union-attr]
    head_weight_before = model.heads["property"].weight.detach().clone()

    loss = model.training_step(batch, batch_idx=0)
    loss.backward()
    optimizer.step()

    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.heads["property"].parameters())
    assert torch.equal(model.backbone.linear.weight, backbone_weight_before)  # type: ignore[union-attr]
    assert not torch.equal(model.heads["property"].weight, head_weight_before)


def test_masked_loss_zeroes_gradients_for_missing_targets() -> None:
    """Missing labels should contribute zero gradient while labeled targets still train."""
    model = MoMLCAModel(property_names=["logS", "logP", "pKa"])
    predictions = torch.tensor(
        [
            [1.5, 1.0, 1.0],
            [1.0, 5.0, 7.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [1.0, float("nan"), 3.0],
            [2.0, 4.0, float("nan")],
        ],
        dtype=torch.float32,
    )

    loss = model._compute_masked_loss(predictions, targets)
    loss.backward()

    assert predictions.grad is not None
    assert torch.isclose(predictions.grad[0, 1], torch.tensor(0.0))
    assert torch.isclose(predictions.grad[1, 2], torch.tensor(0.0))
    assert predictions.grad[0, 0] != 0
    assert predictions.grad[0, 2] != 0
    assert predictions.grad[1, 0] != 0
    assert predictions.grad[1, 1] != 0


def test_task_weights_only_change_masked_loss_not_aggregate_mae() -> None:
    """Task weights should affect the loss while leaving aggregate MAE semantics unchanged."""
    batch = make_batch()
    predictions = torch.tensor(
        [
            [1.5, 1.0, 1.0],
            [1.0, 5.0, 7.0],
        ],
        dtype=torch.float32,
    )
    model = MoMLCAModel(
        property_names=["logS", "logP", "pKa"],
        task_weights={"logS": 2.0, "logP": 1.0, "pKa": 0.5},
    )

    loss = model._compute_masked_loss(predictions, batch.y)
    mae = model._compute_masked_mae(predictions, batch.y)

    expected_weighted_loss = torch.tensor((0.5 + 2.0 + 2.0 + 1.0) / 5.5, dtype=torch.float32)
    expected_unweighted_loss = torch.tensor((0.25 + 4.0 + 1.0 + 1.0) / 4.0, dtype=torch.float32)
    expected_mae = torch.tensor((0.5 + 2.0 + 1.0 + 1.0) / 4.0, dtype=torch.float32)

    assert torch.isclose(loss, expected_weighted_loss)
    assert not torch.isclose(loss, expected_unweighted_loss)
    assert torch.isclose(mae, expected_mae)


def test_task_weight_mapping_must_match_property_names() -> None:
    """Named task weights should fail loudly when they do not align with property names."""
    model = MoMLCAModel(
        property_names=["logS", "logP", "pKa"],
        task_weights={"logS": 1.0, "logP": 2.0},
    )

    with pytest.raises(ValueError, match="task_weights keys must match property_names"):
        model._task_weights_tensor(
            num_targets=3,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_per_task_epoch_metrics_use_valid_label_counts_and_skip_unlabeled_tasks() -> None:
    """Sparse-label epoch metrics should aggregate over labeled entries only."""
    model = MoMLCAModel(property_names=["logS", "logP", "pKa"])
    logged: dict[str, torch.Tensor] = {}
    model.log = lambda name, value, **_: logged.__setitem__(name, value.detach().clone())  # type: ignore[method-assign]

    model.on_train_epoch_start()
    model._update_per_task_metric_state(
        "train",
        predictions=torch.tensor(
            [
                [2.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        targets=torch.tensor(
            [
                [1.0, float("nan"), float("nan")],
                [1.0, float("nan"), float("nan")],
            ],
            dtype=torch.float32,
        ),
    )
    model._update_per_task_metric_state(
        "train",
        predictions=torch.tensor(
            [
                [5.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        targets=torch.tensor(
            [
                [1.0, float("nan"), float("nan")],
                [float("nan"), float("nan"), float("nan")],
            ],
            dtype=torch.float32,
        ),
    )
    model.on_train_epoch_end()

    assert torch.isclose(logged["train/mae_logS"], torch.tensor(2.0))
    assert torch.isclose(logged["train/rmse_logS"], torch.sqrt(torch.tensor(6.0)))
    assert "train/mae_logP" not in logged
    assert "train/rmse_logP" not in logged
    assert model._per_task_metric_state["train"] is None


def test_single_target_regression_uses_generic_metric_names_when_defaults_do_not_fit() -> None:
    """Single-target heads should remain compatible without overriding property names."""
    batch = Batch.from_data_list(
        [
            Data(
                x=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float32),
                y=torch.tensor([[1.0]], dtype=torch.float32),
            ),
            Data(
                x=torch.tensor([[2.0, 1.0]], dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float32),
                y=torch.tensor([[3.0]], dtype=torch.float32),
            ),
        ]
    )
    predictions = torch.tensor([[0.5], [2.0]], dtype=torch.float32)
    model = MoMLCAModel(
        backbone=RecordingBackbone(),
        heads={"property": StaticHead(predictions)},
    )
    logged: list[str] = []
    model.log = lambda name, value, **_: logged.append(name)  # type: ignore[method-assign]

    model.on_train_epoch_start()
    loss = model.training_step(batch, batch_idx=0)
    model.on_train_epoch_end()

    assert loss.shape == torch.Size([])
    assert "train/mae" in logged
    assert "train/mae_target_0" in logged
    assert "train/rmse_target_0" in logged


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
    assert model.backbone.output_dim == 128


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
    assert model.backbone.output_dim == 128


def test_hydra_instantiates_distinct_gin_backbone() -> None:
    """The `model=gin` override should select a concrete backbone config."""
    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=gin", "data=pfasbench"],
        )
        with open_dict(cfg):
            cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))

        model = hydra.utils.instantiate(cfg.model)

    GlobalHydra.instance().clear()

    assert isinstance(model, MoMLCAModel)
    assert isinstance(model.backbone, GINBackbone)
    assert model.backbone.output_dim == 128


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
    assert "train/mae_logS" in metric_dict
    assert "val/mae_logS" in metric_dict


def test_train_fast_dev_run_with_painn_backbone_uses_default_callbacks(tmp_path: Path) -> None:
    """The real training path should also work with `model=painn` in fast-dev mode."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=painn", "data=pfasbench"],
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
            cfg.model.pretrained_backbone.checkpoint_path = None

        HydraConfig().set_config(cfg)
        metric_dict, _ = train(cfg)

    GlobalHydra.instance().clear()

    assert "train/loss" in metric_dict
    assert "val/loss" in metric_dict
    assert "train/mae_logS" in metric_dict
    assert "val/mae_logS" in metric_dict


def test_train_fast_dev_run_with_gin_backbone_uses_default_callbacks(tmp_path: Path) -> None:
    """The real training path should also work with `model=gin` in fast-dev mode."""
    dataset_root = write_sample_pfasbench_dataset(tmp_path / "data")

    with initialize(version_base="1.3", config_path="../../../configs"):
        cfg = compose(
            config_name="config.yaml",
            return_hydra_config=True,
            overrides=["model=gin", "data=pfasbench"],
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
            cfg.model.pretrained_backbone.checkpoint_path = None

        HydraConfig().set_config(cfg)
        metric_dict, _ = train(cfg)

    GlobalHydra.instance().clear()

    assert "train/loss" in metric_dict
    assert "val/loss" in metric_dict
    assert "train/mae_logS" in metric_dict
    assert "val/mae_logS" in metric_dict
