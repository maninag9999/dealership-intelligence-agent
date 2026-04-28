"""
ML Training Pipeline
---------------------
Orchestrates training of all three models:
  1. InventoryAgingModel   (XGBoost regression)
  2. RepClusteringModel    (K-Means archetypes)
  3. CustomerSentimentScorer (DistilBERT — eval only, pre-trained weights)

Data is loaded from DuckDB (the dbt-transformed mart layer).
All results are logged to MLflow.

Run via:
    python scripts/train_models.py
or imported:
    from dealership.ml.train import TrainingPipeline
    results = TrainingPipeline().run()
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import mlflow
import pandas as pd

from dealership.ml.aging_model import InventoryAgingModel
from dealership.ml.rep_clustering import RepClusteringModel
from dealership.ml.sentiment import CustomerSentimentScorer

logger = logging.getLogger(__name__)

# ── Config ───────────────────────────────────────────────────────────────────
DEFAULT_DB_PATH = Path(os.getenv("DEALERSHIP_DB_PATH", "data/dealership.duckdb"))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


# ── SQL queries against dbt mart tables ──────────────────────────────────────
AGING_QUERY = """
SELECT
    make, model, trim, color, fuel_type, transmission,
    year, mileage, price, certified_pre_owned,
    COALESCE(days_since_last_price_drop, 0)  AS days_since_last_price_drop,
    days_on_lot
FROM mart_inventory
WHERE days_on_lot IS NOT NULL
  AND days_on_lot >= 0
"""

REP_QUERY = """
SELECT
    rep_id,
    close_rate,
    avg_deal_value,
    deals_closed,
    avg_days_to_close,
    follow_up_rate,
    customer_satisfaction,
    upsell_rate
FROM mart_sales_reps
"""

SENTIMENT_QUERY = """
SELECT
    review_id,
    review_text,
    CASE WHEN rating >= 4 THEN 1 ELSE 0 END AS true_label
FROM mart_customer_reviews
WHERE review_text IS NOT NULL
  AND LENGTH(TRIM(review_text)) > 5
LIMIT 2000
"""


# ── Results container ─────────────────────────────────────────────────────────
@dataclass
class TrainingResults:
    aging_metrics: dict[str, float] = field(default_factory=dict)
    clustering_metrics: dict[str, float] = field(default_factory=dict)
    sentiment_metrics: dict[str, float] = field(default_factory=dict)
    run_ids: dict[str, str] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return len(self.errors) == 0


# ── Pipeline ─────────────────────────────────────────────────────────────────
class TrainingPipeline:
    """
    End-to-end training pipeline.

    Parameters
    ----------
    db_path : Path to the DuckDB file produced by dbt.
    skip_sentiment : bool — skip DistilBERT eval (useful for fast CI runs).
    """

    def __init__(
        self,
        db_path: Path | str = DEFAULT_DB_PATH,
        skip_sentiment: bool = False,
    ) -> None:
        self.db_path = Path(db_path)
        self.skip_sentiment = skip_sentiment
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    def run(self) -> TrainingResults:
        results = TrainingResults()
        logger.info("=" * 60)
        logger.info("Dealership ML Training Pipeline starting…")
        logger.info("DB: %s", self.db_path.resolve())
        logger.info("=" * 60)

        self._train_aging(results)
        self._train_clustering(results)

        if not self.skip_sentiment:
            self._eval_sentiment(results)
        else:
            logger.info("Skipping sentiment evaluation (skip_sentiment=True).")

        logger.info("=" * 60)
        if results.all_passed:
            logger.info("✅  All models trained successfully.")
        else:
            logger.warning("⚠️  Pipeline finished with errors: %s", results.errors)
        logger.info("=" * 60)

        return results

    # -- private steps --------------------------------------------------------
    def _train_aging(self, results: TrainingResults) -> None:
        logger.info("[1/3] Training InventoryAgingModel (XGBoost)…")
        try:
            df = self._query(AGING_QUERY)
            logger.info("  Loaded %d inventory rows", len(df))

            model = InventoryAgingModel()
            model.fit(df)

            run_id = model.log_to_mlflow()
            results.aging_metrics = model.metrics_
            results.run_ids["aging"] = run_id
            logger.info("  Metrics: %s | run_id=%s", model.metrics_, run_id)
        except Exception as exc:
            msg = f"AgingModel failed: {exc}"
            logger.exception(msg)
            results.errors.append(msg)

    def _train_clustering(self, results: TrainingResults) -> None:
        logger.info("[2/3] Training RepClusteringModel (K-Means)…")
        try:
            df = self._query(REP_QUERY)
            logger.info("  Loaded %d rep rows", len(df))

            model = RepClusteringModel()
            model.fit(df)

            run_id = model.log_to_mlflow()
            results.clustering_metrics = model.metrics_
            results.run_ids["clustering"] = run_id

            logger.info("  Archetypes: %s", model.archetype_labels_)
            logger.info("  Metrics: %s | run_id=%s", model.metrics_, run_id)
            logger.info("\n%s", model.archetype_summary().to_string())
        except Exception as exc:
            msg = f"RepClustering failed: {exc}"
            logger.exception(msg)
            results.errors.append(msg)

    def _eval_sentiment(self, results: TrainingResults) -> None:
        logger.info("[3/3] Evaluating CustomerSentimentScorer (DistilBERT)…")
        try:
            df = self._query(SENTIMENT_QUERY)
            logger.info("  Loaded %d reviews", len(df))

            scorer = CustomerSentimentScorer()
            scorer.load()

            metrics = scorer.evaluate(
                texts=df["review_text"].tolist(),
                true_labels=df["true_label"].tolist(),
            )
            run_id = scorer.log_to_mlflow()
            results.sentiment_metrics = metrics
            results.run_ids["sentiment"] = run_id
            logger.info("  Metrics: %s | run_id=%s", metrics, run_id)
        except Exception as exc:
            msg = f"SentimentScorer failed: {exc}"
            logger.exception(msg)
            results.errors.append(msg)

    def _query(self, sql: str) -> pd.DataFrame:
        """Execute SQL against DuckDB and return a DataFrame."""
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"DuckDB file not found: {self.db_path}\n" "Run dbt models first: `dbt run --project-dir dbt_project`"
            )
        with duckdb.connect(str(self.db_path), read_only=True) as con:
            return con.execute(sql).df()
