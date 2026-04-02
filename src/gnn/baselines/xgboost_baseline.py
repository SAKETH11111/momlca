"""XGBoost baseline utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from gnn.baselines._base import MultiOutputBaselineModel


class XGBoostBaseline(MultiOutputBaselineModel):
    """XGBoost baseline trained on descriptor features."""

    def __init__(
        self,
        property_names: list[str] | None = None,
        feature_names: list[str] | None = None,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_lambda: float = 1.0,
        n_jobs: int = -1,
        random_state: int = 42,
        eval_metric: str = "mae",
        early_stopping_rounds: int | None = 25,
        objective: str = "reg:squarederror",
        tree_method: str = "hist",
        verbosity: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            property_names=property_names,
            feature_names=feature_names,
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stopping_rounds,
            objective=objective,
            tree_method=tree_method,
            verbosity=verbosity,
            **kwargs,
        )

    def _create_model(self) -> Any:
        from xgboost import XGBRegressor

        return XGBRegressor(**self.model_kwargs)

    def _fit_single_target(
        self,
        *,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        if (
            X_val is not None
            and y_val is not None
            and self.model_kwargs.get("early_stopping_rounds") is not None
        ):
            valid_mask = ~np.isnan(y_val)
            if np.any(valid_mask):
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val[valid_mask], y_val[valid_mask])],
                    verbose=False,
                )
                return
        model.fit(X_train, y_train, verbose=False)

    def save(self, path: str | Path) -> None:
        """Persist metadata with native XGBoost model files."""
        path = Path(path)
        base_dir = path if path.suffix == "" else path.with_suffix("")
        base_dir.mkdir(parents=True, exist_ok=True)

        payload = self._serialize_state()
        payload["models"] = list(self._models)
        joblib.dump(payload, base_dir / "metadata.joblib")

        for property_name, model in self._models.items():
            model.get_booster().save_model(base_dir / f"{property_name}.json")

    @classmethod
    def load(cls, path: str | Path) -> XGBoostBaseline:
        """Restore metadata and native XGBoost model files."""
        path = Path(path)
        base_dir = path if path.suffix == "" else path.with_suffix("")
        payload = joblib.load(base_dir / "metadata.joblib")
        instance = cls(
            property_names=payload["property_names"],
            feature_names=payload["feature_names"],
            **payload["model_kwargs"],
        )
        instance._is_fitted = payload["is_fitted"]
        instance._n_features_in = payload["n_features_in"]
        instance._imputation_values = payload["imputation_values"]
        instance._models = {}

        for property_name in payload["models"]:
            model = instance._create_model()
            model.load_model(base_dir / f"{property_name}.json")
            instance._models[property_name] = model

        return instance


def train_xgboost_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    property_names: list[str] | None = None,
    feature_names: list[str] | None = None,
    **model_kwargs: Any,
) -> XGBoostBaseline:
    """Train and return an XGBoost baseline model."""
    model = XGBoostBaseline(
        property_names=property_names,
        feature_names=feature_names,
        **model_kwargs,
    )
    return model.fit(X_train, y_train, X_val=X_val, y_val=y_val)


def predict_xgboost(model: XGBoostBaseline, X: np.ndarray) -> np.ndarray:
    """Generate predictions from a fitted XGBoost baseline."""
    return model.predict(X)


def save_xgboost_model(model: XGBoostBaseline, path: str | Path) -> None:
    """Persist a fitted XGBoost baseline."""
    model.save(path)


def load_xgboost_model(path: str | Path) -> XGBoostBaseline:
    """Load a previously saved XGBoost baseline."""
    return XGBoostBaseline.load(path)


def get_xgboost_feature_importances(model: XGBoostBaseline) -> Any:
    """Return property-wise feature importances."""
    return model.get_feature_importances()


__all__ = [
    "XGBoostBaseline",
    "get_xgboost_feature_importances",
    "load_xgboost_model",
    "predict_xgboost",
    "save_xgboost_model",
    "train_xgboost_baseline",
]
