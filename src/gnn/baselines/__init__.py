"""Traditional descriptor-based baselines for PFAS property prediction."""

from gnn.baselines.comparison import ModelComparison, ModelResult
from gnn.baselines.data_utils import BaselineDataset, extract_baseline_data
from gnn.baselines.descriptors import DescriptorExtractor, MolecularDescriptorExtractor
from gnn.baselines.models import BaselineModel, MultiOutputBaselineModel
from gnn.baselines.random_forest import (
    RandomForestBaseline,
    get_rf_feature_importances,
    load_rf_model,
    predict_rf,
    save_rf_model,
    train_rf_baseline,
)
from gnn.baselines.xgboost_baseline import (
    XGBoostBaseline,
    get_xgboost_feature_importances,
    load_xgboost_model,
    predict_xgboost,
    save_xgboost_model,
    train_xgboost_baseline,
)

__all__ = [
    # Descriptors
    "DescriptorExtractor",
    "MolecularDescriptorExtractor",
    # Models
    "BaselineModel",
    "MultiOutputBaselineModel",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "train_rf_baseline",
    "predict_rf",
    "save_rf_model",
    "load_rf_model",
    "get_rf_feature_importances",
    "train_xgboost_baseline",
    "predict_xgboost",
    "save_xgboost_model",
    "load_xgboost_model",
    "get_xgboost_feature_importances",
    # Data utilities
    "BaselineDataset",
    "extract_baseline_data",
    # Comparison
    "ModelComparison",
    "ModelResult",
]
