"""Physics constraints for charge prediction heads."""

from __future__ import annotations

import torch
from torch import nn


class ChargeConservationLayer(nn.Module):
    """Adjust per-atom charges so each graph sums to its formal charge."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if eps <= 0.0:
            raise ValueError("eps must be positive.")
        self.eps = eps

    def forward(
        self,
        charges: torch.Tensor,
        log_variance: torch.Tensor,
        batch_index: torch.Tensor,
        formal_charges: torch.Tensor | float | int,
    ) -> torch.Tensor:
        """Apply an uncertainty-weighted correction that preserves total charge."""
        if charges.ndim != 1:
            raise ValueError("charges must be a 1D tensor of per-atom predictions.")
        if log_variance.shape != charges.shape:
            raise ValueError("log_variance must have the same shape as charges.")
        if batch_index.shape != charges.shape:
            raise ValueError("batch_index must align with the per-atom charge tensor.")
        if batch_index.dtype != torch.long:
            raise ValueError("batch_index must be a torch.long tensor.")

        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0
        target_total = self._normalize_formal_charges(
            formal_charges=formal_charges,
            num_graphs=num_graphs,
            device=charges.device,
            dtype=charges.dtype,
        )

        current_total = torch.zeros(num_graphs, device=charges.device, dtype=charges.dtype)
        current_total.index_add_(0, batch_index, charges)

        variance = torch.exp(log_variance)
        variance_total = torch.zeros(num_graphs, device=charges.device, dtype=charges.dtype)
        variance_total.index_add_(0, batch_index, variance)
        normalized_weights = variance / (variance_total[batch_index] + self.eps)

        total_delta = target_total - current_total
        return charges + normalized_weights * total_delta[batch_index]

    def _normalize_formal_charges(
        self,
        formal_charges: torch.Tensor | float | int,
        *,
        num_graphs: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if isinstance(formal_charges, torch.Tensor):
            normalized = formal_charges.to(device=device, dtype=dtype).reshape(-1)
        else:
            normalized = torch.tensor([formal_charges], device=device, dtype=dtype)

        if normalized.numel() != num_graphs:
            raise ValueError(
                f"formal_charges must provide exactly {num_graphs} graph-level targets, "
                f"got {normalized.numel()}."
            )
        return normalized


__all__ = ["ChargeConservationLayer"]
