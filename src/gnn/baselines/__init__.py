"""Traditional descriptor-based baselines for PFAS property prediction."""

from gnn.baselines.comparison import ModelComparison, ModelResult
from gnn.baselines.data_utils import BaselineDataset, extract_baseline_data
from gnn.baselines.descriptors import (
    DescriptorExtractor,
    MolecularDescriptorExtractor,
    export_pfasbench_descriptors,
)
from gnn.baselines.models import BaselineModel, MultiOutputBaselineModel
from gnn.baselines.random_forest import (
    RandomForestBaseline,
    get_rf_feature_importances,
    load_rf_model,
    predict_rf,
    save_rf_model,
    train_rf_baseline,
)
from gnn.baselines.random_forest import (
    get_feature_importance as get_rf_feature_importance,
)
from gnn.baselines.xgboost_baseline import (
    XGBoostBaseline,
    get_xgboost_feature_importances,
    load_xgb_model,
    load_xgboost_model,
    predict_xgb,
    predict_xgboost,
    save_xgb_model,
    save_xgboost_model,
    train_xgb_baseline,
    train_xgboost_baseline,
)
from gnn.baselines.xgboost_baseline import (
    get_feature_importance as get_xgb_feature_importance,
)

__all__ = [
    # Descriptors
    "DescriptorExtractor",
    "MolecularDescriptorExtractor",
    "export_pfasbench_descriptors",
    # Models
    "BaselineModel",
    "MultiOutputBaselineModel",
    "RandomForestBaseline",
    "XGBoostBaseline",
    "train_rf_baseline",
    "predict_rf",
    "save_rf_model",
    "load_rf_model",
    "get_rf_feature_importance",
    "get_rf_feature_importances",
    "train_xgb_baseline",
    "train_xgboost_baseline",
    "predict_xgb",
    "predict_xgboost",
    "save_xgb_model",
    "save_xgboost_model",
    "load_xgb_model",
    "load_xgboost_model",
    "get_xgb_feature_importance",
    "get_xgboost_feature_importances",
    # Data utilities
    "BaselineDataset",
    "extract_baseline_data",
    # Comparison
    "ModelComparison",
    "ModelResult",
]
