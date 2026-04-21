"""Tests for graph-level property prediction heads."""

from __future__ import annotations

from uuid import uuid4

import pytest
import torch
from torch import nn

from gnn.models.heads import (
    PropertyHead,
    get_head,
    get_head_class,
    register_head,
    registered_heads,
)


def test_property_head_returns_prediction_tensor_with_configurable_hidden_dims() -> None:
    """The property head should project graph features through an MLP to predictions."""
    head = PropertyHead(
        input_dim=5,
        output_dim=3,
        hidden_dims=(8, 4),
        dropout=0.0,
        activation="relu",
    )
    graph_features = torch.randn(2, 5)

    predictions = head(graph_features)

    assert isinstance(predictions, torch.Tensor)
    assert predictions.shape == (2, 3)


def test_property_head_returns_uncertainty_mapping_when_enabled() -> None:
    """Optional uncertainty output should return both predictions and log-variance."""
    head = PropertyHead(
        input_dim=5,
        output_dim=3,
        hidden_dims=(6,),
        uncertainty=True,
    )
    graph_features = torch.randn(4, 5)

    outputs = head(graph_features)

    assert isinstance(outputs, dict)
    assert set(outputs) == {"predictions", "log_variance"}
    assert outputs["predictions"].shape == (4, 3)
    assert outputs["log_variance"].shape == (4, 3)


def test_property_head_is_registered_for_lookup() -> None:
    """The property head should be discoverable through the local head registry."""
    assert get_head("property") is PropertyHead
    assert registered_heads()["property"] is PropertyHead


def test_head_registry_supports_runtime_extension_registration() -> None:
    """Runtime heads should be registerable via the public registry seam."""
    runtime_name = f"runtime_head_{uuid4().hex}"

    @register_head(runtime_name)
    class _RuntimeHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(5, 1)

        def forward(self, graph_features: torch.Tensor) -> torch.Tensor:
            return self.linear(graph_features)

    assert get_head(runtime_name.upper()) is _RuntimeHead
    assert get_head_class(runtime_name) is _RuntimeHead
    assert registered_heads()[runtime_name] is _RuntimeHead


def test_head_registry_unknown_name_errors_include_available_entries() -> None:
    """Unknown head lookups should provide available names in the exception."""
    with pytest.raises(KeyError) as exc:
        get_head("not_a_real_head")

    message = str(exc.value)
    assert "Available heads:" in message
    assert "property" in message
