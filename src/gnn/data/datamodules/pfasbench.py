"""Lightning DataModule for PFASBench dataset."""

from __future__ import annotations

import logging

import numpy as np
from lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader

from gnn.data.datasets import PFASBenchDataset
from gnn.data.splits import pfas_ood_split, scaffold_split

logger = logging.getLogger(__name__)


class PFASBenchDataModule(LightningDataModule):
    """Lightning DataModule for PFASBench dataset.

    Provides train/val/test dataloaders with configurable split strategies:
    - scaffold: Scaffold-based splitting (default)
    - pfas_ood: PFAS family out-of-distribution splitting
    - random: Random splitting

    Args:
        root: Root directory for dataset storage
        batch_size: Batch size for dataloaders
        split: Split strategy ("scaffold", "pfas_ood", "random")
        seed: Random seed for reproducibility
        num_workers: Number of dataloader workers
        train_frac: Fraction of data for training
        val_frac: Fraction of data for validation
        test_frac: Fraction of data for testing
        holdout: For pfas_ood split: "chain_length" or "headgroup"
        holdout_values: For pfas_ood split: values to hold out for test

    Notes:
        For ``split="pfas_ood"``, the test set membership is determined by the holdout
        logic (family labels), not by ``test_frac``.
    """

    def __init__(
        self,
        root: str = "data/",
        batch_size: int = 32,
        split: str = "scaffold",
        seed: int = 42,
        num_workers: int = 0,
        train_frac: float = 0.8,
        val_frac: float = 0.1,
        test_frac: float = 0.1,
        holdout: str | None = None,
        holdout_values: list[str] | None = None,
    ) -> None:
        """Initialize PFASBenchDataModule.

        Args:
            root: Root directory for dataset storage
            batch_size: Batch size for dataloaders
            split: Split strategy ("scaffold", "pfas_ood", "random")
            seed: Random seed for reproducibility
            num_workers: Number of dataloader workers
            train_frac: Fraction of data for training
            val_frac: Fraction of data for validation
            test_frac: Fraction of data for testing
            holdout: For pfas_ood split: "chain_length" or "headgroup"
            holdout_values: For pfas_ood split: values to hold out for test
        """
        super().__init__()

        # Save hyperparameters for checkpoint compatibility
        self.save_hyperparameters(logger=False)

        # Store parameters
        self.root = root
        self.batch_size = batch_size
        self.split = split
        self.seed = seed
        self.num_workers = num_workers
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.holdout = holdout
        self.holdout_values = holdout_values or []

        # Dataset attributes (initialized in setup)
        self.dataset: PFASBenchDataset | None = None
        self.train_dataset: PFASBenchDataset | None = None
        self.val_dataset: PFASBenchDataset | None = None
        self.test_dataset: PFASBenchDataset | None = None
        self.predict_dataset: PFASBenchDataset | None = None

        self.train_idx: np.ndarray | None = None
        self.val_idx: np.ndarray | None = None
        self.test_idx: np.ndarray | None = None
        self._smiles_cache: list[str] | None = None

    def _validate_split_config(self) -> None:
        if self.split not in ("scaffold", "pfas_ood", "random"):
            raise ValueError(
                f"Invalid split '{self.split}'. Must be one of: 'scaffold', 'pfas_ood', 'random'."
            )

        for name, value in (
            ("train_frac", self.train_frac),
            ("val_frac", self.val_frac),
            ("test_frac", self.test_frac),
        ):
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be between 0 and 1, got {value}")

        total = self.train_frac + self.val_frac + self.test_frac
        if not np.isclose(total, 1.0):
            raise ValueError(f"train_frac + val_frac + test_frac must equal 1.0, got {total:.6f}")

        if self.split == "pfas_ood":
            if self.holdout not in (None, "chain_length", "headgroup"):
                raise ValueError(
                    f"Invalid holdout '{self.holdout}'. Must be 'chain_length' or 'headgroup'."
                )
            if not self.holdout_values:
                raise ValueError(
                    "pfas_ood split requires non-empty holdout_values to form an OOD test set"
                )
            if (self.train_frac + self.val_frac) <= 0.0:
                raise ValueError("pfas_ood split requires train_frac + val_frac > 0")

    def prepare_data(self) -> None:
        """Download and process dataset (called on rank 0 only).

        Lightning ensures this is called only once on a single process,
        making it safe for downloading data.
        """
        # Instantiate dataset to trigger download/processing
        PFASBenchDataset(root=self.root)

    def setup(self, stage: str | None = None) -> None:
        """Set up train/val/test splits.

        This is called on every process in distributed training.
        Loads the dataset and applies the configured split strategy.

        Args:
            stage: Training stage ("fit", "validate", "test", "predict")
        """
        # Load dataset if not already loaded
        if self.dataset is None:
            self.dataset = PFASBenchDataset(root=self.root)

        self._validate_split_config()

        if self.train_idx is None or self.val_idx is None or self.test_idx is None:
            train_idx, val_idx, test_idx = self._compute_split_indices()
            self.train_idx = train_idx
            self.val_idx = val_idx
            self.test_idx = test_idx

        # Stage handling
        if stage in (None, "fit"):
            self.train_dataset = self.dataset[self.train_idx]
            self.val_dataset = self.dataset[self.val_idx]
        elif stage == "validate":
            self.val_dataset = self.dataset[self.val_idx]
        elif stage == "test":
            self.test_dataset = self.dataset[self.test_idx]
        elif stage == "predict":
            self.predict_dataset = self.dataset
        else:
            raise ValueError(f"Unknown stage: {stage}")

        if stage is None:
            self.test_dataset = self.dataset[self.test_idx]

    def _compute_split_indices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized; call setup() first")

        # Extract SMILES strings from dataset for split functions
        if self._smiles_cache is None or len(self._smiles_cache) != len(self.dataset):
            self._smiles_cache = [self.dataset[i].smiles for i in range(len(self.dataset))]
        smiles_list = self._smiles_cache

        if self.split == "scaffold":
            train_idx, val_idx, test_idx = scaffold_split(
                smiles_list,
                seed=self.seed,
                train_frac=self.train_frac,
                val_frac=self.val_frac,
                test_frac=self.test_frac,
            )
            self._log_split_summary(train_idx, val_idx, test_idx)
            return train_idx, val_idx, test_idx
        if self.split == "pfas_ood":
            train_frac = self.train_frac / (self.train_frac + self.val_frac)
            val_frac = self.val_frac / (self.train_frac + self.val_frac)
            train_idx, val_idx, test_idx, stats = pfas_ood_split(
                smiles_list,
                holdout=self.holdout or "chain_length",
                holdout_values=self.holdout_values,
                seed=self.seed,
                train_frac=train_frac,
                val_frac=val_frac,
            )
            logger.info("PFAS OOD split stats: %s", stats)
            self._validate_ood_split_result(train_idx, val_idx, test_idx)
            self._log_split_summary(train_idx, val_idx, test_idx)
            return train_idx, val_idx, test_idx
        train_idx, val_idx, test_idx = self._random_split()
        self._log_split_summary(train_idx, val_idx, test_idx)
        return train_idx, val_idx, test_idx

    def _validate_ood_split_result(
        self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
    ) -> None:
        if len(test_idx) == 0:
            raise ValueError("pfas_ood split produced empty test set; check holdout/holdout_values")

    def _log_split_summary(
        self, train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray
    ) -> None:
        logger.info(
            "Split sizes: train=%d, val=%d, test=%d (strategy=%s)",
            len(train_idx),
            len(val_idx),
            len(test_idx),
            self.split,
        )
        if self.val_frac > 0 and len(val_idx) == 0:
            logger.warning(
                "Validation split is empty (val_frac=%s); this can happen with scaffold grouping on small/low-diversity datasets",
                self.val_frac,
            )
        if self.test_frac > 0 and self.split != "pfas_ood" and len(test_idx) == 0:
            logger.warning(
                "Test split is empty (test_frac=%s); this can happen with scaffold grouping on small/low-diversity datasets",
                self.test_frac,
            )

    def _random_split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform random split of dataset indices.

        Returns:
            Tuple of (train_indices, val_indices, test_indices)
        """
        # Set random seed for reproducibility
        rng = np.random.default_rng(self.seed)

        if self.dataset is None:
            raise RuntimeError("Dataset not initialized; call setup() first")
        n = len(self.dataset)

        # Create shuffled indices
        indices = np.arange(n)
        rng.shuffle(indices)

        train_size = int(self.train_frac * n)
        val_size = int(self.val_frac * n)

        # Split indices
        train_idx = indices[:train_size]
        val_idx = indices[train_size : train_size + val_size]
        test_idx = indices[train_size + val_size :]

        return train_idx, val_idx, test_idx

    def train_dataloader(self) -> PyGDataLoader:
        """Create and return the train dataloader.

        Returns:
            PyG DataLoader for training data
        """
        if self.train_dataset is None:
            raise RuntimeError("train_dataset not set; did you call setup(stage='fit')?")
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> PyGDataLoader:
        """Create and return the validation dataloader.

        Returns:
            PyG DataLoader for validation data
        """
        if self.val_dataset is None:
            raise RuntimeError("val_dataset not set; did you call setup(stage='fit'/'validate')?")
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> PyGDataLoader:
        """Create and return the test dataloader.

        Returns:
            PyG DataLoader for test data
        """
        if self.test_dataset is None:
            raise RuntimeError("test_dataset not set; did you call setup(stage='test')?")
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self) -> PyGDataLoader:
        if self.predict_dataset is None:
            raise RuntimeError("predict_dataset not set; did you call setup(stage='predict')?")
        return PyGDataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
