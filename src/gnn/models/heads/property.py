"""Property prediction heads for graph-level regression."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
from torch import nn

from gnn.models.heads.registry import register_head


def _build_activation(activation: str | nn.Module) -> nn.Module:
    """Create an activation module from a string name or module instance."""
    if isinstance(activation, nn.Module):
        return copy.deepcopy(activation)

    activation_map: dict[str, type[nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "tanh": nn.Tanh,
    }
    normalized_activation = activation.strip().lower()
    activation_cls = activation_map.get(normalized_activation)
    if activation_cls is None:
        supported = ", ".join(sorted(activation_map))
        raise ValueError(
            f"Unsupported activation '{activation}'. Supported activations: {supported}."
        )
    return activation_cls()


@register_head("property")
class PropertyHead(nn.Module):
    """MLP head for graph-level property prediction with optional uncertainty."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.0,
        activation: str | nn.Module = "silu",
        uncertainty: bool = False,
        uncertainty_output: bool | None = None,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        if any(hidden_dim <= 0 for hidden_dim in hidden_dims):
            raise ValueError("hidden_dims must contain only positive integers.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = tuple(hidden_dims)
        self.dropout = dropout
        self.activation = activation
        if uncertainty_output is not None:
            uncertainty = uncertainty_output

        self.uncertainty = uncertainty
        # Backward-compatible alias for older config paths.
        self.uncertainty_output = uncertainty

        hidden_layers: list[nn.Module] = []
        previous_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            hidden_layers.append(nn.Linear(previous_dim, hidden_dim))
            hidden_layers.append(_build_activation(self.activation))
            if self.dropout > 0.0:
                hidden_layers.append(nn.Dropout(self.dropout))
            previous_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*hidden_layers)

        self.prediction_layer = nn.Linear(previous_dim, self.output_dim)
        self.log_variance_layer = (
            nn.Linear(previous_dim, self.output_dim) if self.uncertainty else None
        )

    def forward(self, graph_features: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        """Project graph features to property predictions, optionally with uncertainty."""
        hidden = self.hidden_layers(graph_features)
        predictions = self.prediction_layer(hidden)
        if self.log_variance_layer is None:
            return predictions

        log_variance = self.log_variance_layer(hidden)
        return {
            "predictions": predictions,
            "log_variance": log_variance,
        }


__all__ = ["PropertyHead"]
