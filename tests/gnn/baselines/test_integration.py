"""Integration tests for baseline models with PFASBenchDataModule."""

import shutil
import tempfile
from pathlib import Path

import pytest

from gnn.baselines.data_utils import extract_baseline_data
from gnn.baselines.descriptors import MolecularDescriptorExtractor
from gnn.baselines.models import RandomForestBaseline, XGBoostBaseline
from gnn.data.datamodules import PFASBenchDataModule
from gnn.evaluation.comparison import ModelComparison


@pytest.fixture
def temp_dataset_with_data() -> str:
    """Create temp dataset with sample PFAS data."""
    content = """smiles,name,logS,logP,pKa
C(=O)(C(F)(F)F)O,TFA,-0.5,0.5,0.5
C(=O)(C(C(F)(F)F)(F)F)O,PFPA,-1.0,1.0,0.6
C(=O)(C(C(C(F)(F)F)(F)F)(F)F)O,PFBA,-1.5,1.5,0.7
C(=O)(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)O,PFPeA,-2.0,2.0,0.8
C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHxA,-2.5,2.5,0.9
C(=O)(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O,PFHpA,-3.0,3.0,1.0
c1ccccc1,Benzene,-0.8,2.1,15.0
c1ccc(cc1)O,Phenol,-0.5,1.5,9.9
c1ccc2ccccc2c1,Naphthalene,-3.0,3.3,15.0
CCO,Ethanol,0.5,-0.1,16.0
"""
    temp_dir = tempfile.mkdtemp()
    raw_dir = Path(temp_dir) / "pfasbench" / "raw"
    raw_dir.mkdir(parents=True)
    (raw_dir / "pfasbench.csv").write_text(content)

    yield temp_dir

    shutil.rmtree(temp_dir)


class TestBaselineIntegration:
    """Integration tests for baseline workflow."""

    @pytest.mark.slow
    def test_extract_baseline_data(self, temp_dataset_with_data: str) -> None:
        """Test extracting data from DataModule."""
        dm = PFASBenchDataModule(
            root=temp_dataset_with_data,
            split="random",
            seed=42,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        dm.setup()

        data = extract_baseline_data(dm)

        assert data.X_train.shape[0] > 0
        assert data.y_train.shape[1] == 3  # logS, logP, pKa
        assert data.X_train.shape[1] > 200  # Should have 200+ descriptors
        assert len(data.feature_names) == data.X_train.shape[1]
        assert data.property_names == ["logS", "logP", "pKa"]

    @pytest.mark.slow
    def test_random_forest_workflow(self, temp_dataset_with_data: str) -> None:
        """Test complete Random Forest training workflow."""
        dm = PFASBenchDataModule(
            root=temp_dataset_with_data,
            split="random",
            seed=42,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        dm.setup()

        # Extract features
        data = extract_baseline_data(dm)

        # Train model
        model = RandomForestBaseline(n_estimators=10)
        model.fit(data.X_train, data.y_train)

        # Predict
        predictions = model.predict(data.X_test)

        assert predictions.shape == data.y_test.shape

    @pytest.mark.slow
    def test_xgboost_workflow(self, temp_dataset_with_data: str) -> None:
        """Test complete XGBoost training workflow."""
        dm = PFASBenchDataModule(
            root=temp_dataset_with_data,
            split="random",
            seed=42,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        dm.setup()

        # Extract features
        data = extract_baseline_data(dm)

        # Train model
        model = XGBoostBaseline(n_estimators=10)
        model.fit(data.X_train, data.y_train)

        # Predict
        predictions = model.predict(data.X_test)

        assert predictions.shape == data.y_test.shape

    @pytest.mark.slow
    def test_comparison_workflow(self, temp_dataset_with_data: str) -> None:
        """Test complete comparison workflow with multiple models."""
        dm = PFASBenchDataModule(
            root=temp_dataset_with_data,
            split="random",
            seed=42,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        dm.setup()

        # Extract features
        data = extract_baseline_data(dm)

        # Train models
        rf_model = RandomForestBaseline(n_estimators=10)
        rf_model.fit(data.X_train, data.y_train)

        xgb_model = XGBoostBaseline(n_estimators=10)
        xgb_model.fit(data.X_train, data.y_train)

        # Predict
        rf_preds = rf_model.predict(data.X_test)
        xgb_preds = xgb_model.predict(data.X_test)

        # Compare
        comparison = ModelComparison(property_names=data.property_names)
        comparison.add_result("RandomForest", rf_preds, data.y_test)
        comparison.add_result("XGBoost", xgb_preds, data.y_test)

        # Check results
        df = comparison.to_dataframe()
        assert len(df) == 2
        assert "RandomForest" in df.index
        assert "XGBoost" in df.index

        # Check all metrics are present
        assert "mae_mean" in df.columns
        assert "rmse_mean" in df.columns
        assert "r2_mean" in df.columns
        # Note: spearman_mean requires >2 samples per property, may not be present with small test sets

        # Generate table output
        table = comparison.to_table()
        assert len(table) > 0

    @pytest.mark.slow
    def test_datamodule_not_setup_raises(self, temp_dataset_with_data: str) -> None:
        """Test that extracting from unsetup DataModule raises error."""
        dm = PFASBenchDataModule(
            root=temp_dataset_with_data,
            split="random",
            seed=42,
        )
        # Don't call setup()

        with pytest.raises(RuntimeError, match="not setup"):
            extract_baseline_data(dm)

    @pytest.mark.slow
    def test_custom_extractor(self, temp_dataset_with_data: str) -> None:
        """Test using custom descriptor extractor."""
        dm = PFASBenchDataModule(
            root=temp_dataset_with_data,
            split="random",
            seed=42,
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.2,
        )
        dm.setup()

        # Use custom descriptors
        extractor = MolecularDescriptorExtractor(
            descriptor_names=["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors"]
        )

        data = extract_baseline_data(dm, extractor=extractor)

        assert data.X_train.shape[1] == 5  # Only 5 descriptors
        assert data.feature_names == ["MolWt", "MolLogP", "TPSA", "NumHDonors", "NumHAcceptors"]
