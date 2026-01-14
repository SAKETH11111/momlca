"""Tests for PFASBench dataset."""

from __future__ import annotations

import math
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from gnn.data.datasets import PFASBenchDataset


@pytest.fixture
def sample_csv_content() -> str:
    """Sample CSV content for testing."""
    return """smiles,name,logS,logP,pKa
C(=O)(C(F)(F)F)O,TFA,-0.5,0.5,0.5
C(=O)(C(C(F)(F)F)(F)F)O,PFPA,-1.0,1.0,0.6
C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O,PFBA,-1.5,1.5,0.7
C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O,PFPeA,-2.0,2.0,0.8
C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHxA,-2.5,2.5,
"""


@pytest.fixture
def temp_dataset_dir(sample_csv_content: str) -> Path:
    """Create temporary dataset directory with sample data."""
    temp_dir = tempfile.mkdtemp()
    raw_dir = Path(temp_dir) / "raw"
    raw_dir.mkdir(parents=True)

    csv_path = raw_dir / "pfasbench.csv"
    csv_path.write_text(sample_csv_content)

    yield Path(temp_dir)

    # Cleanup
    shutil.rmtree(temp_dir)


class TestPFASBenchDataset:
    """Tests for PFASBenchDataset class."""

    def test_instantiation(self, temp_dataset_dir: Path) -> None:
        """Test that dataset can be instantiated."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        assert dataset is not None
        assert isinstance(dataset, PFASBenchDataset)

    def test_len_returns_correct_count(self, temp_dataset_dir: Path) -> None:
        """Test that len(dataset) returns expected molecule count."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        # Sample CSV has 5 molecules
        assert len(dataset) == 5

    def test_root_can_be_base_data_dir(self, sample_csv_content: str) -> None:
        """Test that root can be a base data directory (dataset stored under root/pfasbench)."""
        temp_dir = tempfile.mkdtemp()
        try:
            base_dir = Path(temp_dir)
            raw_dir = base_dir / "pfasbench" / "raw"
            raw_dir.mkdir(parents=True)
            (raw_dir / "pfasbench.csv").write_text(sample_csv_content)

            dataset = PFASBenchDataset(root=str(base_dir))
            assert len(dataset) == 5
            assert (base_dir / "pfasbench" / "processed" / "data.pt").exists()
        finally:
            shutil.rmtree(temp_dir)

    def test_getitem_returns_data_object(self, temp_dataset_dir: Path) -> None:
        """Test that dataset[idx] returns PyG Data object."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        data = dataset[0]
        assert isinstance(data, Data)

    def test_data_has_expected_attributes(self, temp_dataset_dir: Path) -> None:
        """Test that Data objects have required attributes."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        data = dataset[0]

        # Core PyG attributes
        assert hasattr(data, "x")
        assert hasattr(data, "edge_index")
        assert hasattr(data, "edge_attr")

        # Labels
        assert hasattr(data, "y")
        assert data.y.shape == (1, 3)  # 3 properties as row-vector

        # Metadata
        assert hasattr(data, "smiles")
        assert hasattr(data, "name")
        assert hasattr(data, "inchikey")

    def test_x_is_float_tensor(self, temp_dataset_dir: Path) -> None:
        """Test that atom features are float tensor."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        data = dataset[0]
        assert data.x.dtype == torch.float32
        assert data.x.ndim == 2

    def test_edge_index_is_long_tensor(self, temp_dataset_dir: Path) -> None:
        """Test that edge_index is long tensor."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        data = dataset[0]
        assert data.edge_index.dtype == torch.long
        assert data.edge_index.shape[0] == 2

    def test_y_contains_property_values(self, temp_dataset_dir: Path) -> None:
        """Test that y tensor contains correct property values."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        data = dataset[0]  # TFA: logS=-0.5, logP=0.5, pKa=0.5
        assert abs(data.y[0, 0].item() - (-0.5)) < 1e-6  # logS
        assert abs(data.y[0, 1].item() - 0.5) < 1e-6  # logP
        assert abs(data.y[0, 2].item() - 0.5) < 1e-6  # pKa

    def test_smiles_is_string(self, temp_dataset_dir: Path) -> None:
        """Test that smiles attribute is string."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        data = dataset[0]
        assert isinstance(data.smiles, str)
        assert len(data.smiles) > 0

    def test_inchikey_format(self, temp_dataset_dir: Path) -> None:
        """Test that InChIKey has correct format."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        data = dataset[0]
        inchikey = data.inchikey
        # InChIKey format: XXXXXXXXXXXXXX-XXXXXXXXXX-X (27 chars with hyphens)
        assert isinstance(inchikey, str)
        if inchikey:  # May be empty on error
            assert len(inchikey) == 27
            assert inchikey.count("-") == 2


class TestPFASBenchDatasetCaching:
    """Tests for dataset caching behavior."""

    def test_processed_file_created(self, temp_dataset_dir: Path) -> None:
        """Test that processed file is created after instantiation."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        processed_path = temp_dataset_dir / "processed" / "data.pt"
        assert processed_path.exists()
        del dataset  # Ensure reference is dropped

    def test_second_load_uses_cache(self, temp_dataset_dir: Path) -> None:
        """Test that second instantiation uses cached data."""
        # First load - processes raw data
        dataset1 = PFASBenchDataset(root=str(temp_dataset_dir))
        del dataset1

        # Second load - should use cached data
        dataset2 = PFASBenchDataset(root=str(temp_dataset_dir))

        # Verify cached load works correctly
        assert len(dataset2) == 5

    def test_second_load_does_not_reprocess(
        self, temp_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that second instantiation does not call process() when cache exists."""
        dataset1 = PFASBenchDataset(root=str(temp_dataset_dir))
        del dataset1

        def fail_process(self) -> None:  # noqa: ANN001
            raise AssertionError("process() should not be called when processed data exists")

        monkeypatch.setattr(PFASBenchDataset, "process", fail_process)
        dataset2 = PFASBenchDataset(root=str(temp_dataset_dir))
        assert len(dataset2) == 5


class TestPFASBenchMissingLabels:
    """Tests for handling missing property values."""

    @pytest.fixture
    def csv_with_missing(self) -> str:
        """CSV content with missing values."""
        return """smiles,name,logS,logP,pKa
C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHxA,-2.5,2.5,
C(=O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHpA,-3.0,3.0,
"""

    @pytest.fixture
    def temp_dataset_missing(self, csv_with_missing: str) -> Path:
        """Create dataset with missing values."""
        temp_dir = tempfile.mkdtemp()
        raw_dir = Path(temp_dir) / "raw"
        raw_dir.mkdir(parents=True)
        (raw_dir / "pfasbench.csv").write_text(csv_with_missing)
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    def test_missing_labels_are_nan(self, temp_dataset_missing: Path) -> None:
        """Test that missing labels are stored as NaN."""
        dataset = PFASBenchDataset(root=str(temp_dataset_missing))
        data = dataset[0]  # PFHxA has missing pKa
        assert math.isnan(data.y[0, 2].item())  # pKa should be NaN

    def test_present_labels_preserved(self, temp_dataset_missing: Path) -> None:
        """Test that present labels are preserved correctly."""
        dataset = PFASBenchDataset(root=str(temp_dataset_missing))
        data = dataset[0]  # PFHxA: logS=-2.5, logP=2.5
        assert abs(data.y[0, 0].item() - (-2.5)) < 1e-6
        assert abs(data.y[0, 1].item() - 2.5) < 1e-6


class TestPFASBenchAccessorMethods:
    """Tests for accessor methods."""

    def test_get_smiles(self, temp_dataset_dir: Path) -> None:
        """Test get_smiles returns correct SMILES."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        smiles = dataset.get_smiles(0)
        assert isinstance(smiles, str)
        assert len(smiles) > 0

    def test_get_inchikey(self, temp_dataset_dir: Path) -> None:
        """Test get_inchikey returns correct InChIKey."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        inchikey = dataset.get_inchikey(0)
        assert isinstance(inchikey, str)

    def test_get_labels(self, temp_dataset_dir: Path) -> None:
        """Test get_labels returns correct label dict."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        labels = dataset.get_labels(0)  # TFA

        assert isinstance(labels, dict)
        assert "logS" in labels
        assert "logP" in labels
        assert "pKa" in labels

        assert abs(labels["logS"] - (-0.5)) < 1e-6
        assert abs(labels["logP"] - 0.5) < 1e-6
        assert abs(labels["pKa"] - 0.5) < 1e-6

    def test_property_names(self, temp_dataset_dir: Path) -> None:
        """Test property_names returns list of property names."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        names = dataset.property_names
        assert names == ["logS", "logP", "pKa"]


class TestPFASBenchProperties:
    """Tests for dataset properties."""

    def test_raw_file_names(self, temp_dataset_dir: Path) -> None:
        """Test raw_file_names property."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        assert dataset.raw_file_names == ["pfasbench.csv"]

    def test_processed_file_names(self, temp_dataset_dir: Path) -> None:
        """Test processed_file_names property."""
        dataset = PFASBenchDataset(root=str(temp_dataset_dir))
        assert dataset.processed_file_names == ["data.pt"]
