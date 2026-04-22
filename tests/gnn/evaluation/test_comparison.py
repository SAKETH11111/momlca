"""Tests for model comparison framework."""

import platform
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gnn.baselines.random_forest import save_rf_model, train_rf_baseline
from gnn.baselines.xgboost_baseline import save_xgb_model, train_xgb_baseline
from gnn.evaluation.comparison import ModelComparison, ModelResult, compute_regression_metrics
from scripts import compare_baselines

XGBOOST_ARTIFACT_LOADING_CRASH_RISK = (
    sys.platform == "darwin" and platform.machine() == "arm64" and sys.version_info >= (3, 12)
)


class TestComputeRegressionMetrics:
    """Test suite for compute_regression_metrics function."""

    def test_perfect_predictions(self) -> None:
        """Test metrics with perfect predictions."""
        y_true = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y_pred = y_true.copy()

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1", "prop2", "prop3"])

        assert metrics["mae_prop1"] == 0.0
        assert metrics["mae_prop2"] == 0.0
        assert metrics["mae_prop3"] == 0.0
        assert metrics["mae_mean"] == 0.0
        assert metrics["rmse_mean"] == 0.0
        assert np.isclose(metrics["r2_mean"], 1.0)

    def test_known_error(self) -> None:
        """Test metrics with known error values."""
        y_true = np.array([[0.0], [1.0], [2.0], [3.0]])
        y_pred = np.array([[0.5], [1.5], [2.5], [3.5]])  # All off by 0.5

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1"])

        assert np.isclose(metrics["mae_prop1"], 0.5)
        assert np.isclose(metrics["rmse_prop1"], 0.5)

    def test_handles_nan_targets(self) -> None:
        """Test that NaN targets are properly masked."""
        y_true = np.array([[1.0, 2.0], [np.nan, 4.0], [3.0, np.nan]])
        y_pred = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 5.0]])

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1", "prop2"])

        # prop1: only indices 0, 2 are valid -> MAE = 0
        assert metrics["mae_prop1"] == 0.0
        # prop2: only indices 0, 1 are valid -> MAE = 0
        assert metrics["mae_prop2"] == 0.0

    def test_1d_input(self) -> None:
        """Test that 1D inputs are handled correctly."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert "mae_prop_0" in metrics
        assert metrics["mae_prop_0"] == 0.0

    def test_spearman_correlation(self) -> None:
        """Test Spearman correlation calculation."""
        # Perfect monotonic relationship
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        y_pred = np.array([[10.0], [20.0], [30.0], [40.0], [50.0]])

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1"])

        assert np.isclose(metrics["spearman_prop1"], 1.0)

    def test_pearson_correlation(self) -> None:
        y_true = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_pred = np.array([[2.0], [4.0], [6.0], [8.0]])

        metrics = compute_regression_metrics(y_true, y_pred, ["prop1"])

        assert np.isclose(metrics["pearson_prop1"], 1.0)


class TestModelComparison:
    """Test suite for ModelComparison class."""

    def test_initialization(self) -> None:
        """Test comparison initialization."""
        comparison = ModelComparison(property_names=["logS", "logP", "pKa"])
        assert comparison.property_names == ["logS", "logP", "pKa"]
        assert len(comparison.results) == 0

    def test_add_result(self) -> None:
        """Test adding model results."""
        comparison = ModelComparison(property_names=["prop1", "prop2"])

        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.1, 2.1], [3.1, 4.1]])

        result = comparison.add_result("Model1", y_pred, y_true)

        assert isinstance(result, ModelResult)
        assert result.model_name == "Model1"
        assert "mae_prop1" in result.metrics
        assert len(comparison.results) == 1

    def test_add_model_then_evaluate(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])

        class ConstantPredictor:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.ones((len(X), 1))

        comparison.add_model("Constant", ConstantPredictor(), model_type="constant")
        results = comparison.evaluate(np.zeros((3, 2)), np.ones((3, 1)))

        assert len(results) == 1
        assert results[0].model_name == "Constant"
        assert results[0].metadata["model_type"] == "constant"

    def test_multiple_models(self) -> None:
        """Test comparing multiple models."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0], [3.0]])

        # Model 1: Good predictions
        comparison.add_result("Good", np.array([[1.0], [2.0], [3.0]]), y_true)

        # Model 2: Poor predictions
        comparison.add_result("Bad", np.array([[0.0], [0.0], [0.0]]), y_true)

        assert len(comparison.results) == 2

        df = comparison.to_dataframe()
        assert len(df) == 2
        assert "Good" in df.index
        assert "Bad" in df.index

        # Good model should have lower MAE
        assert df.loc["Good", "mae_prop1"] < df.loc["Bad", "mae_prop1"]

    def test_to_table(self) -> None:
        """Test table output generation."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [2.0]])

        comparison.add_result("Model1", y_pred, y_true)

        table = comparison.to_table()
        assert "Model1" in table
        assert "mae" in table.lower()

    def test_to_table_filter_metrics(self) -> None:
        """Test filtering metrics in table output."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [2.0]])

        comparison.add_result("Model1", y_pred, y_true)

        table = comparison.to_table(metric_types=["mae", "r2"])
        assert "mae" in table.lower()
        assert "r2" in table.lower()

    def test_multi_split_dataframe_uses_multiindex(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])

        comparison.add_result(
            "Model1", np.array([[1.0], [2.0]]), np.array([[1.0], [2.0]]), split_name="validation"
        )
        comparison.add_result(
            "Model1", np.array([[1.0], [2.0]]), np.array([[1.0], [2.0]]), split_name="test"
        )

        df = comparison.to_dataframe()
        assert ("validation", "Model1") in df.index
        assert ("test", "Model1") in df.index

    def test_save(self) -> None:
        """Test saving comparison to CSV."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        y_pred = np.array([[1.0], [2.0]])

        comparison.add_result("Model1", y_pred, y_true)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.csv"
            comparison.save(str(path))

            assert path.exists()
            content = path.read_text()
            assert "Model1" in content

    def test_save_report(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])
        comparison.add_result("Model1", np.array([[1.0], [2.0]]), np.array([[1.0], [2.0]]))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.md"
            comparison.save_report(path)

            assert path.exists()
            assert "Baseline Model Comparison" in path.read_text()

    def test_get_best_model(self) -> None:
        """Test getting best model."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0], [3.0]])

        # Good model (lower MAE)
        comparison.add_result("Good", np.array([[1.0], [2.0], [3.0]]), y_true)
        # Bad model (higher MAE)
        comparison.add_result("Bad", np.array([[0.0], [0.0], [0.0]]), y_true)

        best = comparison.get_best_model(metric="mae_mean", lower_is_better=True)
        assert best == "Good"

        # For R2, higher is better
        best_r2 = comparison.get_best_model(metric="r2_mean", lower_is_better=False)
        assert best_r2 == "Good"

    def test_get_best_model_invalid_metric(self) -> None:
        """Test that invalid metric raises error."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])
        comparison.add_result("Model1", y_true, y_true)

        with pytest.raises(ValueError, match="not found"):
            comparison.get_best_model(metric="invalid_metric")

    def test_empty_comparison_dataframe(self) -> None:
        """Test that empty comparison returns empty dataframe."""
        comparison = ModelComparison(property_names=["prop1"])
        df = comparison.to_dataframe()
        assert len(df) == 0

    def test_metadata_stored(self) -> None:
        """Test that metadata is stored in results."""
        comparison = ModelComparison(property_names=["prop1"])

        y_true = np.array([[1.0], [2.0]])

        result = comparison.add_result(
            "Model1",
            y_true,
            y_true,
            metadata={"n_estimators": 100, "max_depth": 5},
        )

        assert result.metadata["n_estimators"] == 100
        assert result.metadata["max_depth"] == 5

    def test_to_dataframe_includes_scalar_metadata_columns(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])
        targets = np.array([[1.0], [2.0]])
        comparison.add_result(
            "Model1",
            targets,
            targets,
            metadata={
                "model_type": "gnn",
                "checkpoint_path": "/tmp/checkpoints/best.ckpt",
                "checkpoint_id": "best-abc12345",
                "non_scalar": {"skip": True},
            },
        )

        frame = comparison.to_dataframe()
        assert "checkpoint_path" in frame.columns
        assert "checkpoint_id" in frame.columns
        assert "non_scalar" not in frame.columns
        assert frame.loc["Model1", "checkpoint_id"] == "best-abc12345"

    def test_evaluate_all_splits(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])

        class ConstantPredictor:
            def predict(self, X: np.ndarray) -> np.ndarray:
                return np.ones((len(X), 1))

        comparison.add_model("Constant", ConstantPredictor(), model_type="constant")
        results = comparison.evaluate_all_splits(
            {
                "random": (np.zeros((2, 3)), np.ones((2, 1))),
                "scaffold": (np.zeros((3, 3)), np.ones((3, 1))),
            }
        )

        assert len(results) == 2
        df = comparison.to_dataframe()
        assert ("random", "Constant") in df.index
        assert ("scaffold", "Constant") in df.index

    def test_evaluate_supports_datamodule_predictors(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])

        class DatamodulePredictor:
            def predict_datamodule(
                self, datamodule: SimpleNamespace, *, split_name: str
            ) -> np.ndarray:
                del split_name
                return np.ones_like(datamodule.test_targets)

        datamodule = SimpleNamespace(test_targets=np.ones((3, 1), dtype=float))
        comparison.add_model("Graph", DatamodulePredictor(), model_type="gnn")

        results = comparison.evaluate(
            None,
            datamodule.test_targets,
            split_name="test",
            datamodule=datamodule,
        )

        assert len(results) == 1
        assert results[0].model_name == "Graph"
        assert results[0].metadata["model_type"] == "gnn"

    def test_evaluate_all_splits_supports_datamodule_split_inputs(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])

        class DatamodulePredictor:
            def predict_datamodule(
                self, datamodule: SimpleNamespace, *, split_name: str
            ) -> np.ndarray:
                del split_name
                return np.ones_like(datamodule.test_targets)

        comparison.add_model("Graph", DatamodulePredictor(), model_type="gnn")
        results = comparison.evaluate_all_splits(
            {
                "random": {
                    "targets": np.ones((2, 1), dtype=float),
                    "datamodule": SimpleNamespace(test_targets=np.ones((2, 1), dtype=float)),
                }
            }
        )

        assert len(results) == 1
        assert results[0].split_name == "random"
        df = comparison.to_dataframe()
        assert "Graph" in df.index

    def test_save_report_contains_rankings_and_significance_notes(self) -> None:
        comparison = ModelComparison(property_names=["prop1", "prop2"])
        comparison.add_result(
            "ModelA",
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            split_name="random",
            metadata={"model_type": "rf"},
        )
        comparison.add_result(
            "ModelB",
            np.array([[1.2, 2.2], [2.8, 3.8]]),
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            split_name="scaffold",
            metadata={"model_type": "xgb"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.md"
            comparison.save_report(path)
            content = path.read_text()

        assert "Per-Split Rankings" in content
        assert "Best Model Per Target" in content
        assert "Statistical Significance Notes" in content

    def test_save_report_renders_significance_results_table(self) -> None:
        comparison = ModelComparison(property_names=["prop1"])
        comparison.add_result("A", np.array([[1.0], [2.0]]), np.array([[1.0], [2.0]]))
        comparison.add_result("B", np.array([[1.2], [2.4]]), np.array([[1.0], [2.0]]))
        significance = pd.DataFrame(
            [
                {
                    "model_a": "A",
                    "model_b": "B",
                    "property": "prop1",
                    "metric_proxy": "absolute_error",
                    "sample_count": 2,
                    "test_name": "wilcoxon",
                    "p_value": 0.03,
                    "winning_direction": "A lower absolute_error",
                }
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "comparison.md"
            comparison.save_report(path, significance=significance)
            content = path.read_text()

        assert "Statistical Significance Results" in content
        assert "wilcoxon" in content


class TestCompareBaselinesCliHelpers:
    def test_parse_model_specs_supports_training_and_artifacts(self) -> None:
        specs = compare_baselines.parse_model_specs(
            ["rf", "SavedXGB=xgb:artifacts/xgb_model", "Graph=gnn:checkpoints/best.ckpt"]
        )

        assert [(spec.name, spec.kind, spec.source) for spec in specs] == [
            ("RandomForest", "rf", "train"),
            ("SavedXGB", "xgb", "artifact"),
            ("Graph", "gnn", "artifact"),
        ]
        assert specs[1].path == Path("artifacts/xgb_model")

    def test_parse_model_specs_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            compare_baselines.parse_model_specs(["rf", "RandomForest=rf:models/rf.joblib"])

    @pytest.mark.skipif(
        XGBOOST_ARTIFACT_LOADING_CRASH_RISK,
        reason=(
            "Known XGBoost segmentation fault on macOS arm64 + Python 3.12 during "
            "artifact-loading smoke test; skipped to keep suite stable in this environment."
        ),
    )
    def test_load_artifact_models_supports_rf_and_xgb(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(32, 5))
        y = np.column_stack([X[:, 0] + 0.1 * X[:, 1], X[:, 2] - 0.2 * X[:, 3]])
        feature_names = [f"f{i}" for i in range(X.shape[1])]

        rf_model = train_rf_baseline(
            X,
            y,
            property_names=["logS", "logP"],
            feature_names=feature_names,
            n_estimators=10,
            random_state=7,
        )
        xgb_model = train_xgb_baseline(
            X,
            y[:, 0],
            property_names=["logS"],
            feature_names=feature_names,
            n_estimators=10,
            early_stopping_rounds=None,
            random_state=7,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            rf_path = tmp_path / "rf.joblib"
            xgb_path = tmp_path / "xgb_model"
            save_rf_model(rf_model, rf_path)
            save_xgb_model(xgb_model, xgb_path)

            loaded = compare_baselines.load_artifact_models(
                compare_baselines.parse_model_specs(
                    [f"SavedRF=rf:{rf_path}", f"SavedXGB=xgb:{xgb_path}"]
                )
            )

            assert np.asarray(loaded["SavedRF"].predict(X[:4])).shape == (4, 2)
            assert np.asarray(loaded["SavedXGB"].predict(X[:4])).shape == (4,)

    def test_load_artifact_models_supports_gnn_loader_hook(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class DummyGNNPredictor:
            def __init__(self, path: Path) -> None:
                self.path = path

            def predict_datamodule(
                self, datamodule: SimpleNamespace, *, split_name: str
            ) -> np.ndarray:
                del split_name
                return np.zeros_like(datamodule.test_targets)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "model.ckpt"
            checkpoint_path.write_text("checkpoint")

            monkeypatch.setattr(
                compare_baselines,
                "_import_callable",
                lambda import_path: (lambda path: DummyGNNPredictor(path)),
            )
            loaded = compare_baselines.load_artifact_models(
                compare_baselines.parse_model_specs([f"Graph=gnn:{checkpoint_path}"]),
                gnn_loader="project.module:load_predictor",
            )

            assert isinstance(loaded["Graph"], DummyGNNPredictor)
            assert loaded["Graph"].path == checkpoint_path

    def test_predict_model_prefers_datamodule_interface(self) -> None:
        class DatamodulePredictor:
            def predict_datamodule(
                self, datamodule: SimpleNamespace, *, split_name: str
            ) -> np.ndarray:
                del split_name
                return np.full_like(datamodule.test_targets, 2.0)

        class DatasetPredictor:
            def predict_dataset(self, dataset: SimpleNamespace, *, split_name: str) -> np.ndarray:
                del split_name
                return np.ones_like(dataset.y_test)

        datamodule = SimpleNamespace(test_targets=np.zeros((3, 1), dtype=float))
        dataset = SimpleNamespace(
            X_test=np.zeros((3, 2), dtype=float),
            y_test=np.zeros((3, 1), dtype=float),
        )
        predictions = compare_baselines.predict_model(
            DatamodulePredictor(),
            dataset=dataset,
            datamodule=datamodule,
            split_name="test",
        )

        assert predictions.shape == (3, 1)
        assert np.all(predictions == 2.0)

    def test_predict_model_falls_back_to_dataset_interface(self) -> None:
        class DatasetPredictor:
            def predict_dataset(self, dataset: SimpleNamespace, *, split_name: str) -> np.ndarray:
                del split_name
                return np.ones_like(dataset.y_test)

        dataset = SimpleNamespace(
            X_test=np.zeros((3, 2), dtype=float),
            y_test=np.zeros((3, 1), dtype=float),
        )
        predictions = compare_baselines.predict_model(
            DatasetPredictor(),
            dataset=dataset,
            split_name="test",
        )

        assert predictions.shape == (3, 1)
