"""Tests for node-level charge prediction heads."""

from __future__ import annotations

import torch

from gnn.models.heads import ChargeHead, get_head, registered_heads


def test_charge_head_returns_per_atom_predictions_and_uncertainty() -> None:
    """Charge head should project node embeddings to one charge and uncertainty each."""
    head = ChargeHead(
        input_dim=5,
        hidden_dims=(8, 4),
        dropout=0.0,
        activation="relu",
    )
    node_features = torch.randn(6, 5)

    outputs = head(node_features)

    assert set(outputs) == {"predictions", "log_variance"}
    assert outputs["predictions"].shape == (6,)
    assert outputs["log_variance"].shape == (6,)


def test_charge_head_is_registered_for_lookup() -> None:
    """Charge head should be discoverable through the public head registry."""
    assert get_head("charge") is ChargeHead
    assert registered_heads()["charge"] is ChargeHead
