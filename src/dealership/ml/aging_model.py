"""
Inventory Aging Model
---------------------
XGBoost regressor that predicts days-on-lot for a vehicle listing.
Features: make, model, year, mileage, price, trim, color, fuel_type,
          transmission, certified_pre_owned, days_since_last_price_drop.

MLflow is used for experiment tracking.
SHAP is used for local + global explainability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "inventory_aging_xgb"
EXPERIMENT_NAME = "dealership/inventory_aging"

CATEGORICAL_COLS = [
    "make",
    "model",
    "trim",
    "color",
    "fuel_type",
    "transmission",
]
BOOLEAN_COLS = ["certified_pre_owned"]
NUMERIC_COLS = [
    "year",
    "mileage",
    "price",
    "days_since_last_price_drop",
]

TARGET_COL = "days_on_lot"


# ── Feature engineering ──────────────────────────────────────────────────────
class InventoryFeatureEngineer:
    """Fit-transform style encoder for the aging model feature set."""

    def __init__(self) -> None:
        self._encoders: dict[str, LabelEncoder] = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._encode_categoricals(df, fit=True)
        df = self._engineer(df)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._encode_categoricals(df, fit=False)
        df = self._engineer(df)
        return df

    # -- internal helpers -----------------------------------------------------
    def _encode_categoricals(self, df: pd.DataFrame, *, fit: bool) -> pd.DataFrame:
        for col in CATEGORICAL_COLS:
            if col not in df.columns:
                logger.warning("Column %s missing — filling with 'unknown'", col)
                df[col] = "unknown"
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self._encoders[col] = le
            else:
                le = self._encoders.get(col)
                if le is None:
                    raise RuntimeError(f"Encoder for '{col}' not fitted yet.")
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
                df[col] = le.transform(df[col])
        return df

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        # Vehicle age in years from model year
        current_year = pd.Timestamp.now().year
        if "year" in df.columns:
            df["vehicle_age"] = current_year - df["year"]

        # Price-per-mile ratio (log-scaled to handle zeros)
        if "price" in df.columns and "mileage" in df.columns:
            df["price_per_mile"] = np.log1p(df["price"]) / np.log1p(df["mileage"].clip(lower=1))

        # Boolean cast
        for col in BOOLEAN_COLS:
            if col in df.columns:
                df[col] = df[col].astype(int)

        return df

    def feature_columns(self) -> list[str]:
        return CATEGORICAL_COLS + BOOLEAN_COLS + NUMERIC_COLS + ["vehicle_age", "price_per_mile"]


# ── Model ────────────────────────────────────────────────────────────────────
class InventoryAgingModel:
    """
    XGBoost days-on-lot predictor.

    Usage
    -----
    >>> model = InventoryAgingModel()
    >>> model.fit(df_train)
    >>> preds = model.predict(df_new)
    >>> explanation = model.explain(df_new.head(5))
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = params or self._default_params()
        self.engineer = InventoryFeatureEngineer()
        self.booster: xgb.XGBRegressor | None = None
        self._explainer: shap.TreeExplainer | None = None
        self.metrics_: dict[str, float] = {}

    # -- public API -----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> InventoryAgingModel:
        """Fit on a raw DataFrame that includes TARGET_COL."""
        if TARGET_COL not in df.columns:
            raise ValueError(f"DataFrame must contain '{TARGET_COL}' column.")

        X_raw = df.drop(columns=[TARGET_COL])
        y = df[TARGET_COL].values

        X = self.engineer.fit_transform(X_raw)[self.engineer.feature_columns()]

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        self.booster = xgb.XGBRegressor(**self.params)
        self.booster.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Metrics on validation set
        y_pred = self.booster.predict(X_val)
        self.metrics_ = self._compute_metrics(y_val, y_pred)
        logger.info("Aging model metrics: %s", self.metrics_)

        # Build SHAP explainer once
        self._explainer = shap.TreeExplainer(self.booster)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predicted days-on-lot for each row."""
        self._assert_fitted()
        X = self.engineer.transform(df)[self.engineer.feature_columns()]
        return self.booster.predict(X)  # type: ignore[union-attr]

    def explain(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Return SHAP values + a human-readable feature-importance summary.
        Works for both single rows and batches.
        """
        self._assert_fitted()
        X = self.engineer.transform(df)[self.engineer.feature_columns()]
        shap_values = self._explainer.shap_values(X)  # type: ignore[union-attr]

        importance = (
            pd.Series(
                np.abs(shap_values).mean(axis=0),
                index=self.engineer.feature_columns(),
            )
            .sort_values(ascending=False)
            .to_dict()
        )
        return {
            "shap_values": shap_values,
            "feature_importance": importance,
            "feature_names": self.engineer.feature_columns(),
        }

    def log_to_mlflow(self, artifact_dir: str | Path = "artifacts/aging") -> str:
        """Log params, metrics, and model to MLflow. Returns the run ID."""
        self._assert_fitted()
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=MODEL_NAME) as run:
            mlflow.log_params(self.params)
            mlflow.log_metrics(self.metrics_)
            mlflow.xgboost.log_model(self.booster, artifact_path="model")
            logger.info("Logged aging model to MLflow run %s", run.info.run_id)
            return run.info.run_id

    # -- helpers --------------------------------------------------------------
    def _assert_fitted(self) -> None:
        if self.booster is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

    @staticmethod
    def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))
        return {"rmse": rmse, "mae": mae, "r2": r2}

    @staticmethod
    def _default_params() -> dict[str, Any]:
        return {
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
            "early_stopping_rounds": 20,
        }
