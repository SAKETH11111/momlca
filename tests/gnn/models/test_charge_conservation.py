"""Tests for charge-conservation physics layers."""

from __future__ import annotations

import torch

from gnn.models import ChargeConservationLayer


def test_charge_conservation_layer_enforces_total_charge_per_graph() -> None:
    """Adjusted charges should sum to the requested graph-level formal charge."""
    layer = ChargeConservationLayer()
    charges = torch.tensor([0.2, -0.1, 0.4, 0.3, -0.2], dtype=torch.float32)
    log_variance = torch.log(torch.tensor([0.2, 0.8, 0.4, 0.7, 0.3], dtype=torch.float32))
    batch_index = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    formal_charges = torch.tensor([0.5, -1.0], dtype=torch.float32)

    adjusted = layer(charges, log_variance, batch_index, formal_charges)

    graph_sums = torch.zeros(2, dtype=torch.float32)
    graph_sums.index_add_(0, batch_index, adjusted)
    assert torch.allclose(graph_sums, formal_charges, atol=1e-6)


def test_charge_conservation_layer_assigns_more_correction_to_higher_uncertainty() -> None:
    """Higher uncertainty atoms should absorb more of the conservation correction."""
    layer = ChargeConservationLayer()
    charges = torch.tensor([0.0, 0.0], dtype=torch.float32)
    log_variance = torch.log(torch.tensor([0.1, 0.9], dtype=torch.float32))
    batch_index = torch.tensor([0, 0], dtype=torch.long)

    adjusted = layer(charges, log_variance, batch_index, formal_charges=1.0)

    assert adjusted[1] > adjusted[0]
    assert torch.isclose(adjusted.sum(), torch.tensor(1.0), atol=1e-6)


def test_charge_conservation_layer_rejects_misaligned_graph_targets() -> None:
    """Formal charge targets must match the number of graphs in the batch."""
    layer = ChargeConservationLayer()
    charges = torch.tensor([0.1, -0.1], dtype=torch.float32)
    log_variance = torch.zeros_like(charges)
    batch_index = torch.tensor([0, 1], dtype=torch.long)

    try:
        layer(charges, log_variance, batch_index, formal_charges=torch.tensor([0.0]))
    except ValueError as exc:
        assert "formal_charges must provide exactly 2" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched graph targets.")
