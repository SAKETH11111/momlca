"""3D PaiNN backbone implementation backed by SchNetPack."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool, global_mean_pool

from gnn.data.transforms.constants import ALLOWED_ATOMS, ATOM_ATOMIC_NUMBER_SLICE, NUM_ATOM_TYPES
from gnn.models.backbones.base import BackboneOutput, BaseBackbone
from gnn.models.backbones.registry import register_backbone

_UNKNOWN_ATOMIC_NUMBER = 0


def _resolve_pooling(pooling: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    normalized_pooling = pooling.strip().lower()
    if normalized_pooling == "sum":
        return global_add_pool
    if normalized_pooling == "mean":
        return global_mean_pool
    raise ValueError(f"Unsupported pooling mode '{pooling}'. Supported modes: sum, mean.")


def _load_schnetpack_modules() -> dict[str, Any]:
    try:
        import schnetpack.properties as properties
        from schnetpack.nn import BesselRBF, CosineCutoff, GaussianRBF, MollifierCutoff
        from schnetpack.representation import PaiNN
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "PaiNNBackbone requires SchNetPack. Install it via Poetry dependency `schnetpack`."
        ) from error

    return {
        "properties": properties,
        "PaiNN": PaiNN,
        "GaussianRBF": GaussianRBF,
        "BesselRBF": BesselRBF,
        "CosineCutoff": CosineCutoff,
        "MollifierCutoff": MollifierCutoff,
    }


@register_backbone("painn")
class PaiNNBackbone(BaseBackbone):
    """SchNetPack PaiNN wrapper that conforms to the repo's backbone contract."""

    def __init__(
        self,
        hidden_channels: int = 128,
        num_layers: int = 6,
        cutoff: float = 5.0,
        num_rbf: int = 20,
        radial_basis: str = "gaussian",
        cutoff_fn: str = "cosine",
        rbf_trainable: bool = False,
        pooling: str = "mean",
        epsilon: float = 1e-8,
        shared_interactions: bool = False,
        shared_filters: bool = False,
    ) -> None:
        super().__init__()
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if cutoff <= 0:
            raise ValueError("cutoff must be positive.")
        if num_rbf <= 0:
            raise ValueError("num_rbf must be a positive integer.")

        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.radial_basis = radial_basis.strip().lower()
        self.cutoff_fn = cutoff_fn.strip().lower()
        self.rbf_trainable = rbf_trainable
        self.pooling = pooling.strip().lower()
        self.epsilon = epsilon
        self.shared_interactions = shared_interactions
        self.shared_filters = shared_filters
        self._pool = _resolve_pooling(self.pooling)

        schnetpack_modules = _load_schnetpack_modules()
        properties_module = schnetpack_modules["properties"]
        self._property_z: str = properties_module.Z
        self._property_idx_i: str = properties_module.idx_i
        self._property_idx_j: str = properties_module.idx_j
        self._property_rij: str = properties_module.Rij
        radial_basis_layer = self._build_radial_basis(schnetpack_modules)
        cutoff_layer = self._build_cutoff(schnetpack_modules)
        self.representation = schnetpack_modules["PaiNN"](
            n_atom_basis=self.hidden_channels,
            n_interactions=self.num_layers,
            radial_basis=radial_basis_layer,
            cutoff_fn=cutoff_layer,
            epsilon=self.epsilon,
            shared_interactions=self.shared_interactions,
            shared_filters=self.shared_filters,
        )

    @property
    def output_dim(self) -> int:
        """Return graph-feature dimension used by downstream heads."""
        return self.hidden_channels

    def _build_radial_basis(self, schnetpack_modules: dict[str, Any]) -> torch.nn.Module:
        if self.radial_basis == "gaussian":
            return schnetpack_modules["GaussianRBF"](
                n_rbf=self.num_rbf,
                cutoff=self.cutoff,
                trainable=self.rbf_trainable,
            )
        if self.radial_basis == "bessel":
            return schnetpack_modules["BesselRBF"](
                n_rbf=self.num_rbf,
                cutoff=self.cutoff,
            )
        raise ValueError(
            f"Unsupported radial_basis '{self.radial_basis}'. Supported values: gaussian, bessel."
        )

    def _build_cutoff(self, schnetpack_modules: dict[str, Any]) -> torch.nn.Module:
        if self.cutoff_fn == "cosine":
            return schnetpack_modules["CosineCutoff"](self.cutoff)
        if self.cutoff_fn == "mollifier":
            return schnetpack_modules["MollifierCutoff"](self.cutoff)
        raise ValueError(
            f"Unsupported cutoff_fn '{self.cutoff_fn}'. Supported values: cosine, mollifier."
        )

    def _extract_atomic_numbers(self, batch: Batch) -> torch.Tensor:
        if batch.x.shape[-1] < NUM_ATOM_TYPES:
            raise ValueError(
                "PaiNNBackbone expected atom features to include atomic-number one-hot "
                f"at indices [0:{NUM_ATOM_TYPES}], but got shape {tuple(batch.x.shape)}."
            )

        atomic_one_hot = batch.x[:, ATOM_ATOMIC_NUMBER_SLICE]
        if atomic_one_hot.shape[-1] != NUM_ATOM_TYPES:
            raise ValueError(
                "PaiNNBackbone could not resolve atomic-number feature slice for this batch."
            )

        if not torch.all((atomic_one_hot >= -1e-6) & (atomic_one_hot <= 1.0 + 1e-6)):
            raise ValueError(
                "PaiNNBackbone expected normalized one-hot atomic-number features in batch.x."
            )

        row_sums = atomic_one_hot.sum(dim=-1)
        max_values, max_indices = atomic_one_hot.max(dim=-1)
        expected_ones = torch.ones_like(row_sums)
        if not torch.allclose(row_sums, expected_ones, atol=1e-4, rtol=0):
            raise ValueError(
                "PaiNNBackbone could not infer atomic numbers: atomic-number features are not "
                "one-hot encoded per node."
            )
        if not torch.allclose(max_values, torch.ones_like(max_values), atol=1e-4, rtol=0):
            raise ValueError(
                "PaiNNBackbone could not infer atomic numbers: atomic-number features are ambiguous."
            )

        allowed_atomic_numbers = torch.tensor(
            [*ALLOWED_ATOMS, _UNKNOWN_ATOMIC_NUMBER],
            device=batch.x.device,
            dtype=torch.long,
        )
        atomic_numbers = allowed_atomic_numbers[max_indices]
        if torch.any(atomic_numbers == _UNKNOWN_ATOMIC_NUMBER):
            raise ValueError(
                "PaiNNBackbone cannot map 'other' atomic-number category to a concrete "
                "nuclear charge for SchNetPack inputs."
            )
        return atomic_numbers

    def _build_schnetpack_inputs(
        self, batch: Batch, atomic_numbers: torch.Tensor
    ) -> dict[str, Any]:
        edge_index = batch.edge_index
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "PaiNNBackbone expects edge_index with shape [2, num_edges] in each batch."
            )

        idx_i = edge_index[1].long()
        idx_j = edge_index[0].long()
        r_ij = batch.pos[idx_j] - batch.pos[idx_i]

        if r_ij.shape[-1] != 3:
            raise ValueError("PaiNNBackbone expects 3D edge vectors with shape [num_edges, 3].")

        return {
            self._property_z: atomic_numbers.long(),
            self._property_idx_i: idx_i,
            self._property_idx_j: idx_j,
            self._property_rij: r_ij,
        }

    def forward(self, batch: Batch) -> BackboneOutput:
        """Encode a molecular graph batch into equivariant node and invariant graph features."""
        if getattr(batch, "pos", None) is None:
            raise ValueError("PaiNNBackbone requires 3D positions (`batch.pos`) for each node.")
        if batch.pos.ndim != 2 or batch.pos.shape[-1] != 3:
            raise ValueError(
                f"PaiNNBackbone expected batch.pos with shape [num_nodes, 3], got {tuple(batch.pos.shape)}."
            )

        atomic_numbers = self._extract_atomic_numbers(batch)
        schnetpack_inputs = self._build_schnetpack_inputs(batch, atomic_numbers)
        schnetpack_outputs = self.representation(schnetpack_inputs)
        scalar_features = schnetpack_outputs["scalar_representation"]
        vector_features = schnetpack_outputs["vector_representation"]
        batch_index = getattr(batch, "batch", None)
        if batch_index is None:
            batch_index = torch.zeros(batch.x.shape[0], dtype=torch.long, device=batch.x.device)

        graph_features = self._pool(scalar_features, batch_index)

        return {
            "node_features": vector_features,
            "graph_features": graph_features,
        }


__all__ = ["PaiNNBackbone"]
