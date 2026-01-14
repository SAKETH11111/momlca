"""PFASBench dataset for PFAS property prediction benchmarking.

This module provides the PFASBenchDataset class, a PyTorch Geometric InMemoryDataset
for training and evaluating GNN models on PFAS molecular property prediction tasks.
"""

from __future__ import annotations

import logging
import os.path as osp
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, InMemoryDataset

from gnn.data.loaders import load_sdf, load_smiles
from gnn.data.transforms import mol_to_pyg_data
from gnn.exceptions import FeaturizationError, InvalidSMILESError

logger = logging.getLogger(__name__)


class PFASBenchDataset(InMemoryDataset):
    """PFASBench: A benchmark dataset for PFAS property prediction.

    This dataset provides curated PFAS (per- and polyfluoroalkyl substances)
    molecules with property labels for training and evaluating molecular
    property prediction models.

    Properties:
        - logS: Aqueous solubility (log mol/L)
        - logP: Octanol-water partition coefficient
        - pKa: Acid dissociation constant

    Args:
        root: Root directory where the dataset should be stored.
            The raw data should be in ``root/raw/`` and processed data
            will be saved to ``root/processed/``.
        transform: Optional transform applied to each Data object
            when accessed via ``__getitem__``.
        pre_transform: Optional transform applied to each Data object
            during processing (before saving).
        pre_filter: Optional filter predicate applied during processing.
            Data objects for which the filter returns False are excluded.

    Example:
        >>> dataset = PFASBenchDataset(root="data/pfasbench")
        >>> len(dataset)
        14
        >>> data = dataset[0]
        >>> data.smiles
        'O=C(O)C(F)(F)F'
        >>> dataset.get_labels(0)
        {'logS': -0.5, 'logP': 0.5, 'pKa': 0.5}
    """

    url: str | None = None  # Placeholder for download URL

    def __init__(
        self,
        root: str,
        transform: Callable[[Data], Data] | None = None,
        pre_transform: Callable[[Data], Data] | None = None,
        pre_filter: Callable[[Data], bool] | None = None,
    ) -> None:
        resolved_root = self._resolve_dataset_root(root)
        super().__init__(resolved_root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])
        self._upgrade_labels_if_needed()

    def _upgrade_labels_if_needed(self) -> None:
        """Upgrade cached labels to row-vector format if needed.

        Older cached datasets stored per-graph ``y`` as shape ``(num_properties,)``,
        which batches into a flattened 1D tensor. The current format stores
        per-graph labels as ``(1, num_properties)`` so batches yield
        ``(batch_size, num_properties)``.
        """
        data = getattr(self, "data", None)
        slices = getattr(self, "slices", None)
        if data is None or slices is None:
            return

        y = getattr(data, "y", None)
        y_slices = slices.get("y") if isinstance(slices, dict) else None
        if y is None or y_slices is None:
            return
        if not isinstance(y, torch.Tensor) or not isinstance(y_slices, torch.Tensor):
            return

        if y.ndim != 1:
            return

        num_graphs = int(y_slices.numel()) - 1
        num_props = len(self.property_names)
        if num_graphs <= 0 or num_props <= 0:
            return
        if int(y_slices[-1].item()) != int(y.numel()):
            return

        segment_sizes = (y_slices[1:] - y_slices[:-1]).tolist()
        if not all(int(size) == num_props for size in segment_sizes):
            return

        data.y = y.view(num_graphs, num_props)
        slices["y"] = torch.arange(num_graphs + 1, dtype=torch.long)

    @staticmethod
    def _resolve_dataset_root(root: str) -> str:
        """Resolve the on-disk dataset root from a user-provided root path.

        This supports two common calling patterns:
        - ``root="data/pfasbench"`` (dataset root)
        - ``root="data"`` (base data dir; dataset stored in ``data/pfasbench``)
        """
        root_path = Path(root)
        if root_path.name == "pfasbench":
            return str(root_path)
        if (root_path / "raw").exists() or (root_path / "processed").exists():
            return str(root_path)
        return str(root_path / "pfasbench")

    @property
    def raw_file_names(self) -> list[str]:
        """List of raw files required for this dataset."""
        return ["pfasbench.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        """List of processed files created by this dataset."""
        return ["data.pt"]

    @property
    def property_names(self) -> list[str]:
        """List of property names in the dataset."""
        return ["logS", "logP", "pKa"]

    @property
    def property_units(self) -> dict[str, str]:
        """Units metadata for each property."""
        return {
            "logS": "log(mol/L)",
            "logP": "unitless",
            "pKa": "pH",
        }

    def download(self) -> None:
        """Download raw data files.

        Currently a placeholder for network download, but supports local
        conversion from an SDF if provided.

        If ``pfasbench.csv`` is missing but ``pfasbench.sdf`` exists, this
        generates ``pfasbench.csv`` by extracting SMILES and any present
        properties as SD tags.
        """
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        csv_path = raw_dir / "pfasbench.csv"
        if csv_path.exists():
            return

        sdf_path = raw_dir / "pfasbench.sdf"
        if sdf_path.exists():
            molecules = load_sdf(sdf_path)
            records: list[dict[str, object]] = []
            for mol in molecules:
                smiles = Chem.MolToSmiles(mol, canonical=True)
                if mol.HasProp("name"):
                    name = mol.GetProp("name")
                elif mol.HasProp("_Name"):
                    name = mol.GetProp("_Name")
                else:
                    name = ""

                row: dict[str, object] = {
                    "smiles": smiles,
                    "name": name,
                }
                for prop in self.property_names:
                    if mol.HasProp(prop):
                        row[prop] = mol.GetProp(prop)
                records.append(row)

            pd.DataFrame.from_records(records).to_csv(csv_path, index=False)
            logger.info("Generated %s from %s", csv_path, sdf_path)
            return

        raise FileNotFoundError(
            f"PFASBench raw data not found. Place 'pfasbench.csv' (or 'pfasbench.sdf') in {raw_dir}"
        )

    def process(self) -> None:
        """Process raw data into PyG Data objects and save to disk."""
        raw_path = osp.join(self.raw_dir, "pfasbench.csv")
        if not osp.exists(raw_path):
            raise FileNotFoundError(
                f"Missing raw file {raw_path}. Place pfasbench.csv in {self.raw_dir} (or provide pfasbench.sdf)."
            )
        df = pd.read_csv(raw_path)
        if "smiles" not in df.columns:
            raise ValueError(f"Missing required column 'smiles' in {raw_path}")

        data_list: list[Data] = []
        skipped = 0

        for idx, row in df.iterrows():
            try:
                smiles_raw = row["smiles"]
                if pd.isna(smiles_raw) or not str(smiles_raw).strip():
                    raise InvalidSMILESError("Empty SMILES string")
                input_smiles = str(smiles_raw).strip()
                mol = load_smiles(input_smiles)
            except InvalidSMILESError as e:
                logger.warning("Skipping molecule %d (%s): %s", idx, row.get("smiles"), e)
                skipped += 1
                continue

            try:
                data = mol_to_pyg_data(mol, include_pos=False)
            except FeaturizationError as e:
                logger.warning("Skipping molecule %d (%s): %s", idx, row.get("smiles"), e)
                skipped += 1
                continue

            # Add property labels as tensor (NaN preserved for missing values)
            y_values = []
            for prop in self.property_names:
                val = row.get(prop, float("nan"))
                if pd.isna(val):
                    y_values.append(float("nan"))
                else:
                    try:
                        y_values.append(float(val))
                    except (TypeError, ValueError):
                        logger.warning(
                            "Invalid %s value for molecule %d (%s): %r; storing NaN",
                            prop,
                            idx,
                            row.get("smiles"),
                            val,
                        )
                        y_values.append(float("nan"))
            # Store as row-vector so batched graph labels become (batch_size, num_properties)
            data.y = torch.tensor([y_values], dtype=torch.float32)

            # Add metadata
            data.smiles = Chem.MolToSmiles(mol, canonical=True)
            name = row.get("name", "")
            data.name = "" if pd.isna(name) else str(name)
            try:
                data.inchikey = Chem.MolToInchiKey(mol)
            except Exception:
                data.inchikey = ""

            # Apply pre_filter and pre_transform
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        if skipped > 0:
            logger.info(
                "Processed %d molecules, skipped %d due to errors",
                len(data_list),
                skipped,
            )

        self.save(data_list, self.processed_paths[0])

    def get_smiles(self, idx: int) -> str:
        """Get SMILES string for molecule at index.

        Args:
            idx: Index of molecule in dataset.

        Returns:
            SMILES string representation.
        """
        return str(self[idx].smiles)

    def get_inchikey(self, idx: int) -> str:
        """Get InChIKey for molecule at index.

        Args:
            idx: Index of molecule in dataset.

        Returns:
            InChIKey string identifier.
        """
        return str(self[idx].inchikey)

    def get_labels(self, idx: int) -> dict[str, float]:
        """Get property labels for molecule at index.

        Args:
            idx: Index of molecule in dataset.

        Returns:
            Dictionary mapping property names to values.
            Values may be NaN for missing properties.
        """
        y = self[idx].y.squeeze(0)
        return {name: y[i].item() for i, name in enumerate(self.property_names)}
