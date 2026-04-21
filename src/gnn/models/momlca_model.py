"""Base LightningModule for MoMLCA graph regression training."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from hydra.utils import to_absolute_path
from lightning import LightningModule
from torch import nn
from torch_geometric.data import Batch

from gnn.models.backbones.base import BackboneOutput, BaseBackbone
from gnn.models.heads import PropertyHead

DEFAULT_PROPERTY_NAMES: tuple[str, ...] = ("logS", "logP", "pKa")


def _supports_weights_only(callable_obj: Callable[..., Any]) -> bool:
    """Return whether ``callable_obj`` accepts the ``weights_only`` kwarg."""
    return "weights_only" in inspect.signature(callable_obj).parameters


class _MeanPoolBackbone(BaseBackbone):
    """Fallback backbone that mean-pools node features into graph features."""

    def __init__(self, output_dim: int = 128) -> None:
        super().__init__()
        self._output_dim = output_dim
        self.node_projection = nn.LazyLinear(output_dim)
        self.activation = nn.SiLU()

    @property
    def output_dim(self) -> int:
        """Return graph-feature dimension used by downstream heads."""
        return self._output_dim

    def forward(self, batch: Batch) -> BackboneOutput:
        node_features = self.activation(self.node_projection(batch.x))
        graph_features = []
        for graph_index in range(batch.num_graphs):
            mask = batch.batch == graph_index
            graph_features.append(node_features[mask].mean(dim=0))
        return {
            "node_features": node_features,
            "graph_features": torch.stack(graph_features, dim=0),
        }


class MoMLCAModel(LightningModule):
    """Reusable LightningModule shell for PFASBench graph regression."""

    def __init__(
        self,
        backbone: BaseBackbone | None = None,
        heads: Mapping[str, nn.Module] | nn.ModuleDict | None = None,
        optimizer: Callable[..., torch.optim.Optimizer] | None = None,
        scheduler: Callable[..., Any] | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        property_names: Sequence[str] | None = None,
        task_weights: Mapping[str, float] | Sequence[float] | None = None,
        pretrained_backbone: Mapping[str, Any] | None = None,
        compile: bool = False,
    ) -> None:
        """Initialize the model shell.

        Args:
            backbone: Graph encoder implementing the shared ``BaseBackbone`` interface.
            heads: Prediction heads keyed by name.
            optimizer: Optional Hydra-instantiated optimizer partial.
            scheduler: Optional Hydra-instantiated scheduler partial.
            learning_rate: Default AdamW learning rate.
            weight_decay: Default AdamW weight decay.
            property_names: Ordered PFASBench regression targets.
            task_weights: Optional per-property loss weights.
            pretrained_backbone: Optional pretrained-backbone loading configuration.
            compile: Whether to compile modules during fit setup.
        """
        super().__init__()
        self.save_hyperparameters(logger=False)

        self._property_names_provided = property_names is not None
        self.property_names = (
            list(property_names) if property_names is not None else list(DEFAULT_PROPERTY_NAMES)
        )
        if backbone is not None and not isinstance(backbone, BaseBackbone):
            raise TypeError(
                "MoMLCAModel requires `backbone` to implement BaseBackbone; "
                f"received {type(backbone).__name__}."
            )
        self.backbone = backbone if backbone is not None else _MeanPoolBackbone()
        self.heads = self._normalize_heads(heads)
        self.optimizer_factory = optimizer
        self.scheduler_factory = scheduler
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.task_weights_config = task_weights
        self.pretrained_backbone_config = self._normalize_pretrained_backbone_config(
            pretrained_backbone
        )
        self.compile_model = compile
        self._is_compiled = False
        self._pretrained_backbone_loaded = False
        self._per_task_metric_state: dict[str, dict[str, Any] | None] = {
            "train": None,
            "val": None,
            "test": None,
        }
        self._load_pretrained_backbone_if_configured()

    def forward(self, batch: Batch) -> dict[str, Any]:
        """Run the batch through the backbone, then through all configured heads."""
        backbone_outputs = self._ensure_backbone_outputs(self.backbone(batch), batch)
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
            self.backbone = cast(BaseBackbone, torch.compile(self.backbone))
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

    def on_train_epoch_start(self) -> None:
        """Reset aggregated per-task training metrics for the new epoch."""
        self._reset_per_task_metric_state("train")

    def on_validation_epoch_start(self) -> None:
        """Reset aggregated per-task validation metrics for the new epoch."""
        self._reset_per_task_metric_state("val")

    def on_test_epoch_start(self) -> None:
        """Reset aggregated per-task test metrics for the new epoch."""
        self._reset_per_task_metric_state("test")

    def on_train_epoch_end(self) -> None:
        """Log exact per-task training metrics accumulated across the epoch."""
        self._log_per_task_epoch_metrics("train")

    def on_validation_epoch_end(self) -> None:
        """Log exact per-task validation metrics accumulated across the epoch."""
        self._log_per_task_epoch_metrics("val")

    def on_test_epoch_end(self) -> None:
        """Log exact per-task test metrics accumulated across the epoch."""
        self._log_per_task_epoch_metrics("test")

    def configure_optimizers(self) -> Any:
        """Create the optimizer and optional scheduler for Lightning."""
        trainable_parameters = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        optimizer_factory = self.optimizer_factory
        if optimizer_factory is not None:
            optimizer = optimizer_factory(params=trainable_parameters)
        else:
            optimizer = torch.optim.AdamW(
                trainable_parameters,
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
        self._update_per_task_metric_state(stage, predictions, batch.y)

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

    def _select_property_predictions(self, predictions: Mapping[str, Any]) -> torch.Tensor:
        if "property" in predictions:
            return self._extract_prediction_tensor(predictions["property"])
        if len(predictions) == 1:
            return self._extract_prediction_tensor(next(iter(predictions.values())))
        raise ValueError(
            "MoMLCAModel requires a 'property' head or exactly one prediction head for regression."
        )

    def _extract_prediction_tensor(self, head_output: Any) -> torch.Tensor:
        """Normalize a head output into the tensor used for loss/metric computation."""
        if isinstance(head_output, torch.Tensor):
            return head_output
        if isinstance(head_output, Mapping):
            log_variance = head_output.get("log_variance")
            if log_variance is not None and not isinstance(log_variance, torch.Tensor):
                raise ValueError(
                    "Head output mappings may include a tensor-valued 'log_variance' entry."
                )
            predictions = head_output.get("predictions")
            if isinstance(predictions, torch.Tensor):
                return predictions
            raise ValueError(
                "Head output mappings must include a tensor-valued 'predictions' entry."
            )
        raise ValueError(
            "Head outputs must be either tensors or mappings with a tensor-valued "
            "'predictions' entry."
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

    def _ensure_backbone_outputs(
        self,
        backbone_outputs: Any,
        batch: Batch,
    ) -> dict[str, Any]:
        """Validate backbone outputs against the shared node+graph contract."""
        if isinstance(backbone_outputs, Mapping):
            normalized_outputs: dict[str, Any] = dict(backbone_outputs)
            if "node_features" not in normalized_outputs:
                raise ValueError(
                    "Backbone outputs must include a tensor-valued 'node_features' entry."
                )
            if "graph_features" not in normalized_outputs:
                raise ValueError(
                    "Backbone outputs must include a tensor-valued 'graph_features' entry."
                )

            node_features = normalized_outputs["node_features"]
            graph_features = normalized_outputs["graph_features"]
            if not isinstance(node_features, torch.Tensor):
                raise ValueError(
                    "Backbone outputs must include a tensor-valued 'node_features' entry."
                )
            if not isinstance(graph_features, torch.Tensor):
                raise ValueError(
                    "Backbone outputs must include a tensor-valued 'graph_features' entry."
                )

            if node_features.shape[0] != batch.x.shape[0]:
                raise ValueError(
                    "Backbone output 'node_features' must align with the batch node count."
                )
            if graph_features.shape[0] != batch.num_graphs:
                raise ValueError(
                    "Backbone output 'graph_features' must align with the batch graph count."
                )
            if graph_features.shape[-1] != self.backbone.output_dim:
                raise ValueError(
                    "Backbone output 'graph_features' last dimension "
                    f"({graph_features.shape[-1]}) does not match "
                    f"backbone.output_dim ({self.backbone.output_dim})."
                )
            return normalized_outputs

        raise ValueError(
            "Backbone must return a mapping with tensor-valued 'node_features' and "
            "'graph_features' entries."
        )

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
            normalized_heads["property"] = PropertyHead(
                input_dim=self.backbone.output_dim,
                output_dim=len(self.property_names),
            )

        return normalized_heads

    def _normalize_pretrained_backbone_config(
        self, pretrained_backbone: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        if pretrained_backbone is None:
            return {
                "checkpoint_path": None,
                "checkpoint_format": "lightning",
                "backbone_key_prefix": "backbone.",
                "freeze_backbone": False,
            }

        normalized_config = dict(pretrained_backbone)
        normalized_config.setdefault("checkpoint_path", None)
        normalized_config.setdefault("checkpoint_format", "lightning")
        normalized_config.setdefault("backbone_key_prefix", "backbone.")
        normalized_config.setdefault("freeze_backbone", False)
        return normalized_config

    def _load_pretrained_backbone_if_configured(self) -> None:
        checkpoint_path = self.pretrained_backbone_config.get("checkpoint_path")
        if checkpoint_path in (None, ""):
            return

        checkpoint = self._load_checkpoint_payload(self._resolve_checkpoint_path(checkpoint_path))
        checkpoint_state = self._extract_checkpoint_state_dict(checkpoint)
        backbone_state = self._extract_backbone_state_dict(checkpoint_state)
        incompatible_keys = self.backbone.load_state_dict(backbone_state, strict=False)
        missing_keys = sorted(incompatible_keys.missing_keys)
        unexpected_keys = sorted(incompatible_keys.unexpected_keys)
        if missing_keys or unexpected_keys:
            raise ValueError(
                "Pretrained backbone checkpoint is incompatible with the current backbone "
                f"(missing={missing_keys}, unexpected={unexpected_keys})"
            )

        self._pretrained_backbone_loaded = True
        if bool(self.pretrained_backbone_config.get("freeze_backbone", False)):
            self._freeze_backbone()

    def _resolve_checkpoint_path(self, checkpoint_path: Any) -> Path:
        normalized_path = Path(str(checkpoint_path)).expanduser()
        if normalized_path.is_absolute():
            return normalized_path
        return Path(to_absolute_path(str(normalized_path)))

    def _load_checkpoint_payload(self, checkpoint_path: Path) -> Any:
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrained backbone checkpoint does not exist: {checkpoint_path}"
            )

        load_kwargs: dict[str, Any] = {"map_location": "cpu"}
        checkpoint_format = str(
            self.pretrained_backbone_config.get("checkpoint_format", "lightning")
        )
        if checkpoint_format == "state_dict" and _supports_weights_only(torch.load):
            load_kwargs["weights_only"] = True
        elif checkpoint_format == "lightning" and _supports_weights_only(torch.load):
            # Lightning checkpoints may include non-tensor Python objects and must
            # only be loaded from trusted sources.
            load_kwargs["weights_only"] = False
        return torch.load(checkpoint_path, **load_kwargs)

    def _extract_checkpoint_state_dict(self, checkpoint: Any) -> Mapping[str, Any]:
        checkpoint_format = str(
            self.pretrained_backbone_config.get("checkpoint_format", "lightning")
        )
        if checkpoint_format not in {"lightning", "state_dict"}:
            raise ValueError(
                "pretrained_backbone.checkpoint_format must be either 'lightning' or 'state_dict'"
            )

        if checkpoint_format == "lightning":
            state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, Mapping) else None
            if not isinstance(state_dict, Mapping):
                raise ValueError(
                    "Lightning checkpoints must contain a top-level 'state_dict' mapping"
                )
            return state_dict

        if isinstance(checkpoint, Mapping) and isinstance(checkpoint.get("state_dict"), Mapping):
            return cast(Mapping[str, Any], checkpoint["state_dict"])
        if not isinstance(checkpoint, Mapping):
            raise ValueError("State-dict checkpoints must deserialize to a mapping of tensors")
        return cast(Mapping[str, Any], checkpoint)

    def _extract_backbone_state_dict(
        self,
        checkpoint_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        prefixes = self._normalize_backbone_prefixes(
            self.pretrained_backbone_config.get("backbone_key_prefix", "backbone.")
        )

        for prefix in prefixes:
            prefix_text = str(prefix)
            backbone_state = {
                key[len(prefix_text) :]: value
                for key, value in checkpoint_state.items()
                if key.startswith(prefix_text)
            }
            if backbone_state:
                return backbone_state

        raise ValueError(
            "Checkpoint did not contain any backbone parameters after applying the configured "
            f"prefix mapping: {prefixes}"
        )

    def _normalize_backbone_prefixes(self, prefix_config: Any) -> list[str]:
        if prefix_config is None:
            return [""]
        if isinstance(prefix_config, str):
            return [prefix_config]
        if isinstance(prefix_config, Sequence):
            prefixes = [str(prefix) for prefix in prefix_config]
            if not prefixes:
                return [""]
            return prefixes
        raise ValueError(
            "pretrained_backbone.backbone_key_prefix must be a string, sequence of strings, or null"
        )

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)

    def _compute_masked_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self._masked_reduce(predictions, targets, reduction="mse", apply_task_weights=True)

    def _compute_masked_mae(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self._masked_reduce(predictions, targets, reduction="mae", apply_task_weights=False)

    def _compute_per_task_error_sums(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        predictions_matrix, targets_matrix = self._ensure_target_matrix(predictions, targets)
        property_names = self._resolve_property_names(num_targets=targets_matrix.shape[-1])
        valid_mask = (~torch.isnan(targets_matrix)) & (~torch.isnan(predictions_matrix))
        safe_targets = torch.nan_to_num(targets_matrix, nan=0.0)
        errors = predictions_matrix - safe_targets

        abs_error_sum = torch.where(valid_mask, errors.abs(), 0.0).sum(dim=0)
        sq_error_sum = torch.where(valid_mask, errors.pow(2), 0.0).sum(dim=0)
        counts = valid_mask.sum(dim=0).to(dtype=predictions_matrix.dtype)
        return property_names, abs_error_sum, sq_error_sum, counts

    def _reset_per_task_metric_state(self, stage: str) -> None:
        self._per_task_metric_state[stage] = None

    def _update_per_task_metric_state(
        self,
        stage: str,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        property_names, abs_error_sum, sq_error_sum, counts = self._compute_per_task_error_sums(
            predictions,
            targets,
        )
        state = self._per_task_metric_state.get(stage)
        if state is None:
            self._per_task_metric_state[stage] = {
                "property_names": property_names,
                "abs_error_sum": abs_error_sum.detach().clone(),
                "sq_error_sum": sq_error_sum.detach().clone(),
                "counts": counts.detach().clone(),
            }
            return

        if state["property_names"] != property_names:
            raise ValueError("Per-task metric state requires stable property names within an epoch")
        state["abs_error_sum"] = cast(torch.Tensor, state["abs_error_sum"]) + abs_error_sum.detach()
        state["sq_error_sum"] = cast(torch.Tensor, state["sq_error_sum"]) + sq_error_sum.detach()
        state["counts"] = cast(torch.Tensor, state["counts"]) + counts.detach()

    def _log_per_task_epoch_metrics(self, stage: str) -> None:
        state = self._per_task_metric_state.get(stage)
        if state is None:
            return

        property_names = cast(list[str], state["property_names"])
        abs_error_sum = cast(torch.Tensor, state["abs_error_sum"])
        sq_error_sum = cast(torch.Tensor, state["sq_error_sum"])
        counts = cast(torch.Tensor, state["counts"])

        for index, property_name in enumerate(property_names):
            if not torch.gt(counts[index], 0).item():
                continue
            count = counts[index]
            self.log(
                f"{stage}/mae_{property_name}",
                abs_error_sum[index] / count,
                on_step=False,
                on_epoch=True,
            )
            self.log(
                f"{stage}/rmse_{property_name}",
                torch.sqrt(torch.clamp(sq_error_sum[index] / count, min=0.0)),
                on_step=False,
                on_epoch=True,
            )

        self._per_task_metric_state[stage] = None

    def _masked_reduce(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str,
        apply_task_weights: bool,
    ) -> torch.Tensor:
        predictions_matrix, targets_matrix = self._ensure_target_matrix(predictions, targets)
        num_targets = targets_matrix.shape[-1]
        valid_mask = (~torch.isnan(targets_matrix)) & (~torch.isnan(predictions_matrix))
        if not torch.any(valid_mask):
            return predictions.sum() * 0.0

        errors = predictions_matrix - torch.nan_to_num(targets_matrix, nan=0.0)
        if reduction == "mse":
            errors = errors.pow(2)
        elif reduction == "mae":
            errors = errors.abs()
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        weights = valid_mask.to(dtype=predictions_matrix.dtype)
        if apply_task_weights:
            task_weights = self._task_weights_tensor(
                num_targets=num_targets,
                device=predictions_matrix.device,
                dtype=predictions_matrix.dtype,
            )
            if task_weights is not None:
                weights = weights * task_weights.unsqueeze(0)

        weighted_error = errors * weights
        denominator = weights.sum()
        if torch.isclose(denominator, torch.zeros_like(denominator)):
            return predictions_matrix.sum() * 0.0
        return weighted_error.sum() / denominator

    def _ensure_target_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if predictions.shape != targets.shape:
            raise ValueError(
                "predictions and targets must have the same shape "
                f"({tuple(predictions.shape)} != {tuple(targets.shape)})"
            )

        if predictions.ndim == 1:
            return predictions.unsqueeze(-1), targets.unsqueeze(-1)

        return predictions, targets

    def _resolve_property_names(self, num_targets: int) -> list[str]:
        if self._property_names_provided and len(self.property_names) != num_targets:
            raise ValueError("property_names must match the number of prediction targets")

        if self._property_names_provided:
            return list(self.property_names)
        if num_targets == len(DEFAULT_PROPERTY_NAMES):
            return list(DEFAULT_PROPERTY_NAMES)
        return [f"target_{index}" for index in range(num_targets)]

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
            property_names = self._resolve_property_names(num_targets)
            provided_keys = set(task_weights)
            expected_keys = set(property_names)
            if provided_keys != expected_keys:
                missing = sorted(expected_keys - provided_keys)
                unexpected = sorted(provided_keys - expected_keys)
                details = []
                if missing:
                    details.append(f"missing: {missing}")
                if unexpected:
                    details.append(f"unexpected: {unexpected}")
                suffix = f" ({'; '.join(details)})" if details else ""
                raise ValueError(f"task_weights keys must match property_names exactly{suffix}")
            values = [
                float(task_weights.get(property_name, 1.0)) for property_name in property_names
            ]
        else:
            values = [float(value) for value in task_weights]
            if len(values) != num_targets:
                raise ValueError("task_weights length must match the number of prediction targets")
        return torch.tensor(values, device=device, dtype=dtype)


__all__ = ["MoMLCAModel"]
