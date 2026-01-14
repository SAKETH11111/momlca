"""Tests for PFASBenchDataModule."""

import shutil
import tempfile
from pathlib import Path

import hydra
import pytest
import rootutils
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict
from torch_geometric.data import Batch

from gnn.data.datamodules import PFASBenchDataModule


@pytest.fixture
def sample_csv_content() -> str:
    """Sample CSV content for testing."""
    return """smiles,name,logS,logP,pKa
C(=O)(C(F)(F)F)O,TFA,-0.5,0.5,0.5
C(=O)(C(C(F)(F)F)(F)F)O,PFPA,-1.0,1.0,0.6
C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O,PFBA,-1.5,1.5,0.7
C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O,PFPeA,-2.0,2.0,0.8
C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHxA,-2.5,2.5,0.9
C(=O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHpA,-3.0,3.0,1.0
C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFOA,-3.5,3.5,1.1
c1ccccc1,Benzene,-0.8,2.1,15.0
c1ccc(cc1)O,Phenol,-0.5,1.5,9.9
c1ccc2ccccc2c1,Naphthalene,-3.0,3.3,15.0
CC(C)c1ccc(cc1)C(C)C,p-Cymene,-2.5,4.1,15.0
c1ccc(cc1)C(=O)O,BenzoicAcid,-1.1,1.9,4.2
c1ccc(cc1)N,Aniline,-0.9,0.9,4.6
c1ccccc1F,Fluorobenzene,-2.3,2.3,15.0
"""


@pytest.fixture
def temp_dataset_dir(sample_csv_content: str) -> Path:
    """Create temporary dataset directory with sample data."""
    temp_dir = tempfile.mkdtemp()
    raw_dir = Path(temp_dir) / "pfasbench" / "raw"
    raw_dir.mkdir(parents=True)

    csv_path = raw_dir / "pfasbench.csv"
    csv_path.write_text(sample_csv_content)

    yield Path(temp_dir)

    # Cleanup
    shutil.rmtree(temp_dir)


class TestPFASBenchDataModule:
    """Test suite for PFASBenchDataModule."""

    def test_instantiation_with_defaults(self):
        """Test that DataModule can be instantiated with default parameters."""
        dm = PFASBenchDataModule()
        assert dm.root == "data/"
        assert dm.batch_size == 32
        assert dm.split == "scaffold"
        assert dm.seed == 42
        assert dm.num_workers == 0
        assert dm.train_frac == 0.8
        assert dm.val_frac == 0.1
        assert dm.test_frac == 0.1
        assert dm.holdout is None
        assert dm.holdout_values == []

    def test_instantiation_with_custom_params(self):
        """Test instantiation with custom parameters."""
        dm = PFASBenchDataModule(
            root="custom_data/",
            batch_size=64,
            split="pfas_ood",
            seed=123,
            num_workers=4,
            train_frac=0.7,
            val_frac=0.15,
            test_frac=0.15,
            holdout="chain_length",
            holdout_values=["C8", "C10"],
        )
        assert dm.root == "custom_data/"
        assert dm.batch_size == 64
        assert dm.split == "pfas_ood"
        assert dm.seed == 123
        assert dm.num_workers == 4
        assert dm.train_frac == 0.7
        assert dm.val_frac == 0.15
        assert dm.test_frac == 0.15
        assert dm.holdout == "chain_length"
        assert dm.holdout_values == ["C8", "C10"]

    def test_prepare_data_creates_dataset(self, temp_dataset_dir: Path):
        """Test that prepare_data creates the dataset."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir))
        # This should trigger dataset download/processing
        dm.prepare_data()
        # Dataset directory should exist after prepare_data
        assert temp_dataset_dir.exists()
        # Processed data should be created
        assert (temp_dataset_dir / "pfasbench" / "processed").exists()

    def test_setup_with_scaffold_split(self, temp_dataset_dir: Path):
        """Test setup with scaffold split strategy."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), split="scaffold", seed=42)
        dm.setup()

        # Check datasets are created
        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None

        # Check that splits sum to total dataset size
        total_size = len(dm.train_dataset) + len(dm.val_dataset) + len(dm.test_dataset)
        assert total_size == len(dm.dataset)

        # Train should always have samples
        assert len(dm.train_dataset) > 0

    def test_setup_with_pfas_ood_split(self, temp_dataset_dir: Path):
        """Test setup with PFAS OOD split strategy."""
        dm = PFASBenchDataModule(
            root=str(temp_dataset_dir),
            split="pfas_ood",
            holdout="chain_length",
            holdout_values=["C8"],
            seed=42,
        )
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
        assert len(dm.test_dataset) > 0

    def test_setup_with_random_split(self, temp_dataset_dir: Path):
        """Test setup with random split strategy."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), split="random", seed=42)
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
        assert len(dm.test_dataset) > 0
        assert dm.train_idx is not None
        assert dm.val_idx is not None
        assert dm.test_idx is not None

    def test_setup_stage_fit_then_test(self, temp_dataset_dir: Path):
        """Test setup respects stage parameter."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), split="scaffold", seed=42)
        dm.setup(stage="fit")

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is None

        dm.setup(stage="test")
        assert dm.test_dataset is not None

    def test_train_dataloader_returns_pyg_dataloader(self, temp_dataset_dir: Path):
        """Test that train_dataloader returns a PyG DataLoader."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), batch_size=32)
        dm.setup()

        train_loader = dm.train_dataloader()
        assert train_loader is not None
        assert hasattr(train_loader, "__iter__")

    def test_val_dataloader_returns_pyg_dataloader(self, temp_dataset_dir: Path):
        """Test that val_dataloader returns a PyG DataLoader."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), batch_size=32)
        dm.setup()

        val_loader = dm.val_dataloader()
        assert val_loader is not None
        assert hasattr(val_loader, "__iter__")

    def test_test_dataloader_returns_pyg_dataloader(self, temp_dataset_dir: Path):
        """Test that test_dataloader returns a PyG DataLoader."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), batch_size=32)
        dm.setup()

        test_loader = dm.test_dataloader()
        assert test_loader is not None
        assert hasattr(test_loader, "__iter__")

    def test_dataloaders_require_setup(self):
        """Test that dataloaders raise if setup hasn't been called."""
        dm = PFASBenchDataModule()
        with pytest.raises(RuntimeError, match="setup"):
            dm.train_dataloader()
        with pytest.raises(RuntimeError, match="setup"):
            dm.val_dataloader()
        with pytest.raises(RuntimeError, match="setup"):
            dm.test_dataloader()

    def test_predict_dataloader_requires_setup(self):
        """Test that predict_dataloader requires setup(stage='predict')."""
        dm = PFASBenchDataModule()
        with pytest.raises(RuntimeError, match="setup"):
            dm.predict_dataloader()

    def test_predict_dataloader_returns_pyg_dataloader(self, temp_dataset_dir: Path):
        """Test predict_dataloader returns a PyG DataLoader."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), batch_size=32)
        dm.setup(stage="predict")

        predict_loader = dm.predict_dataloader()
        assert predict_loader is not None
        assert hasattr(predict_loader, "__iter__")

    def test_batch_has_correct_structure(self, temp_dataset_dir: Path):
        """Test that batches from dataloaders have correct PyG structure."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), batch_size=4)
        dm.setup()

        train_loader = dm.train_dataloader()
        batch = next(iter(train_loader))

        # Batch should be a PyG Batch object
        assert isinstance(batch, Batch)

        # Check required attributes
        assert hasattr(batch, "x")  # Node features
        assert hasattr(batch, "edge_index")  # Edge connectivity
        assert hasattr(batch, "edge_attr")  # Bond features
        assert hasattr(batch, "y")  # Labels
        assert hasattr(batch, "batch")  # Batch assignment vector

        # Check shapes
        assert batch.x is not None
        assert batch.edge_index is not None
        assert batch.edge_attr is not None
        assert batch.y is not None
        assert batch.batch is not None

        # Node features should be 2D
        assert len(batch.x.shape) == 2
        # Edge index should be 2 x num_edges
        assert batch.edge_index.shape[0] == 2
        # Edge attributes should be 2D
        assert len(batch.edge_attr.shape) == 2
        # Labels should be 2D: (batch_size, num_properties)
        assert len(batch.y.shape) == 2
        assert batch.y.shape[0] == batch.num_graphs

    def test_reproducibility_with_same_seed(self, temp_dataset_dir: Path):
        """Test that same seed produces same splits."""
        dm1 = PFASBenchDataModule(root=str(temp_dataset_dir), split="scaffold", seed=42)
        dm1.setup()

        dm2 = PFASBenchDataModule(root=str(temp_dataset_dir), split="scaffold", seed=42)
        dm2.setup()

        assert dm1.train_idx is not None
        assert dm2.train_idx is not None
        assert (dm1.train_idx == dm2.train_idx).all()
        assert (dm1.val_idx == dm2.val_idx).all()
        assert (dm1.test_idx == dm2.test_idx).all()

    def test_different_seeds_produce_different_splits(self, temp_dataset_dir: Path):
        """Test that different seeds produce different splits for random split."""
        dm1 = PFASBenchDataModule(root=str(temp_dataset_dir), split="random", seed=42)
        dm1.setup()

        dm2 = PFASBenchDataModule(root=str(temp_dataset_dir), split="random", seed=123)
        dm2.setup()

        assert dm1.train_idx is not None
        assert dm2.train_idx is not None
        assert not (dm1.train_idx == dm2.train_idx).all()

    def test_hyperparameters_saved(self, temp_dataset_dir: Path):
        """Test that hyperparameters are saved via save_hyperparameters."""
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), batch_size=64, seed=99)
        assert hasattr(dm, "hparams")
        assert dm.hparams.batch_size == 64
        assert dm.hparams.seed == 99

    def test_hydra_config_instantiation(self):
        """Test that Hydra configs can instantiate the DataModule."""
        with initialize(version_base="1.3", config_path="../../../../configs"):
            for data_cfg in (
                "pfasbench",
                "pfasbench_scaffold",
                "pfasbench_ood_chain",
                "pfasbench_ood_headgroup",
            ):
                cfg = compose(
                    config_name="train.yaml",
                    return_hydra_config=True,
                    overrides=[f"data={data_cfg}"],
                )
                with open_dict(cfg):
                    cfg.paths.root_dir = str(rootutils.find_root(indicator=".project-root"))
                datamodule = hydra.utils.instantiate(cfg.data)
                assert isinstance(datamodule, PFASBenchDataModule)

        GlobalHydra.instance().clear()

    def test_invalid_split_raises(self, temp_dataset_dir: Path):
        dm = PFASBenchDataModule(root=str(temp_dataset_dir), split="nope")
        with pytest.raises(ValueError, match="Invalid split"):
            dm.setup()

    def test_pfas_ood_requires_holdout_values(self, temp_dataset_dir: Path):
        dm = PFASBenchDataModule(
            root=str(temp_dataset_dir),
            split="pfas_ood",
            holdout="chain_length",
            holdout_values=[],
        )
        with pytest.raises(ValueError, match="holdout_values"):
            dm.setup()

    def test_pfas_ood_requires_non_empty_test_set(self, temp_dataset_dir: Path):
        dm = PFASBenchDataModule(
            root=str(temp_dataset_dir),
            split="pfas_ood",
            holdout="chain_length",
            holdout_values=["C999"],
        )
        with pytest.raises(ValueError, match="empty test set"):
            dm.setup()
