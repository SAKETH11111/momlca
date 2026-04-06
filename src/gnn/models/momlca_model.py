"""Base LightningModule for MoMLCA graph regression training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
from lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch

DEFAULT_PROPERTY_NAMES: tuple[str, ...] = ("logS", "logP", "pKa")


class _MeanPoolBackbone(nn.Module):
    """Fallback backbone that mean-pools node features into graph features."""

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(batch.x[mask].mean(dim=0))
        return {"graph_features": torch.stack(graph_features, dim=0)}


class _LinearPropertyHead(nn.Module):
    """Fallback property head that lazily projects graph features to targets."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.proj = nn.LazyLinear(output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)


class MoMLCAModel(LightningModule):
    """Reusable LightningModule shell for PFASBench graph regression."""

    def __init__(
        self,
        backbone: nn.Module | None = None,
        heads: Mapping[str, nn.Module] | nn.ModuleDict | None = None,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        scheduler: Callable[..., Any] | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        property_names: Sequence[str] | None = None,
        task_weights: Mapping[str, float] | Sequence[float] | None = None,
        compile: bool = False,
    ) -> None:
        """Initialize the model shell.

        Args:
            backbone: Graph encoder receiving a PyG ``Batch``.
            heads: Prediction heads keyed by name.
            optimizer: Optional Hydra-instantiated optimizer partial.
            scheduler: Optional Hydra-instantiated scheduler partial.
            learning_rate: Default AdamW learning rate.
            weight_decay: Default AdamW weight decay.
            property_names: Ordered PFASBench regression targets.
            task_weights: Optional per-property loss weights.
            compile: Whether to compile modules during fit setup.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.property_names = (
            list(property_names) if property_names is not None else list(DEFAULT_PROPERTY_NAMES)
        )
        self.backbone = backbone if backbone is not None else _MeanPoolBackbone()
        self.heads = self._normalize_heads(heads)
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task_weights_config = task_weights
        self.compile_model = compile
        self._is_compiled = False

    def forward(self, batch: Batch) -> dict[str, Any]:
        """Run the batch through the backbone, then through all configured heads."""
        backbone_outputs = self.backbone(batch)
        predictions = {
            head_name: head(self._resolve_head_inputs(head_name, backbone_outputs))
            for head_name, head in self.heads.items()
        }
        return {
            "backbone": backbone_outputs,
            "predictions": predictions,
        }

    def setup(self, stage: str) -> None:
        """Compile the backbone and heads when requested for fit."""
        if self.compile_model and stage == "fit" and not self._is_compiled:
            self.backbone = cast(nn.Module, torch.compile(self.backbone))
            for head_name, head in list(self.heads.items()):
                self.heads[head_name] = cast(nn.Module, torch.compile(head))
            self._is_compiled = True

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        """Compute and log masked regression metrics for a training batch."""
        loss, metrics = self._shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        """Compute and log masked regression metrics for a validation batch."""
        self._shared_step(batch, stage="val")

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        """Compute and log masked regression metrics for a test batch."""
        self._shared_step(batch, stage="test")

    def configure_optimizers(self) -> Any:
        """Create the optimizer and optional scheduler for Lightning."""
        optimizer_factory = self.optimizer_factory
        if optimizer_factory is not None:
            optimizer = optimizer_factory(params=self.parameters())
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

        scheduler_factory = self.scheduler_factory
        if scheduler_factory is None:
            return {"optimizer": optimizer}

        scheduler = scheduler_factory(optimizer=optimizer)
        scheduler_config: dict[str, Any] = {
            "scheduler": scheduler,
            "interval": "epoch",
            "frequency": 1,
        }
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_config["monitor"] = "val/loss"
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }

    def _shared_step(
        self, batch: Batch, stage: str
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        outputs = self.forward(batch)
        predictions = self._select_property_predictions(outputs["predictions"])
        loss = self._compute_masked_loss(predictions, batch.y)
        mae = self._compute_masked_mae(predictions, batch.y)

        metrics = {"loss": loss, "mae": mae}
        for metric_name, metric_value in metrics.items():
            self.log(
                f"{stage}/{metric_name}",
                metric_value,
                on_step=False,
                on_epoch=True,
                prog_bar=metric_name == "loss",
                batch_size=batch.num_graphs,
            )
        return loss, metrics

    def _select_property_predictions(self, predictions: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if "property" in predictions:
            return predictions["property"]
        if len(predictions) == 1:
            return next(iter(predictions.values()))
        raise ValueError(
            "MoMLCAModel requires a 'property' head or exactly one prediction head for regression."
        )

    def _resolve_head_inputs(self, head_name: str, backbone_outputs: Any) -> Any:
        if isinstance(backbone_outputs, Mapping):
            head_inputs = backbone_outputs.get("head_inputs")
            if isinstance(head_inputs, Mapping) and head_name in head_inputs:
                return head_inputs[head_name]
            if head_name in backbone_outputs:
                return backbone_outputs[head_name]
            if "graph_features" in backbone_outputs:
                return backbone_outputs["graph_features"]
        return backbone_outputs

    def _normalize_heads(
        self, heads: Mapping[str, nn.Module] | nn.ModuleDict | None
    ) -> nn.ModuleDict:
        if heads is None:
            normalized_heads = nn.ModuleDict()
        elif isinstance(heads, nn.ModuleDict):
            normalized_heads = heads
        else:
            normalized_heads = nn.ModuleDict(dict(heads))

        if len(normalized_heads) == 0:
            normalized_heads["property"] = _LinearPropertyHead(
                output_dim=max(len(self.property_names), 1)
            )

        return normalized_heads

    def _compute_masked_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self._masked_reduce(predictions, targets, reduction="mse")

    def _compute_masked_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._masked_reduce(predictions, targets, reduction="mae")

    def _masked_reduce(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        valid_mask = (~torch.isnan(targets)) & (~torch.isnan(predictions))
        if not torch.any(valid_mask):
            return predictions.sum() * 0.0

        errors = predictions - torch.nan_to_num(targets, nan=0.0)
        if reduction == "mse":
            errors = errors.pow(2)
        elif reduction == "mae":
            errors = errors.abs()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        weights = valid_mask.to(dtype=predictions.dtype)
        task_weights = self._task_weights_tensor(
            num_targets=targets.shape[-1],
            device=predictions.device,
            dtype=predictions.dtype,
        )
        if task_weights is not None:
            weights = weights * task_weights.unsqueeze(0)

        weighted_error = errors * weights
        denominator = weights.sum()
        if torch.isclose(denominator, torch.zeros_like(denominator)):
            return predictions.sum() * 0.0
        return weighted_error.sum() / denominator

    def _task_weights_tensor(
        self,
        num_targets: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        task_weights = self.task_weights_config
        if task_weights is None:
            return None

        if isinstance(task_weights, Mapping):
            if self.property_names and len(self.property_names) != num_targets:
                raise ValueError("property_names must match the number of prediction targets")
            property_names = self.property_names or [
                f"target_{index}" for index in range(num_targets)
            ]
            values = [
                float(task_weights.get(property_name, 1.0)) for property_name in property_names
            ]
        else:
            values = [float(value) for value in task_weights]
            if len(values) != num_targets:
                raise ValueError("task_weights length must match the number of prediction targets")
        return torch.tensor(values, device=device, dtype=dtype)


__all__ = ["MoMLCAModel"]
