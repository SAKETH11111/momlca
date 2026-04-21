"""Tests for graph-level property prediction heads."""

from __future__ import annotations

import torch

from gnn.models.heads import PropertyHead, get_head, registered_heads


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
