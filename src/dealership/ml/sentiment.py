"""
Customer Satisfaction Sentiment Scorer
---------------------------------------
Uses DistilBERT (distilbert-base-uncased-finetuned-sst-2-english) to
score customer review text as POSITIVE / NEGATIVE with a confidence score.

For the portfolio demo this model runs fully locally via HuggingFace
Transformers — no API calls, no cost.

The class also exposes:
  - batch_score()  : efficient batched inference for large review tables
  - score_df()     : convenience wrapper that adds columns to a DataFrame
  - evaluate()     : compute precision/recall/F1 if ground-truth labels exist
"""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import pandas as pd
from sklearn.metrics import classification_report

logger = logging.getLogger(__name__)

MODEL_CHECKPOINT = "distilbert-base-uncased-finetuned-sst-2-english"
EXPERIMENT_NAME = "dealership/sentiment"
DEFAULT_BATCH_SIZE = 32
MAX_LENGTH = 512


# ── Sentiment scorer ─────────────────────────────────────────────────────────
class CustomerSentimentScorer:
    """
    DistilBERT-powered sentiment scorer for customer reviews.

    Usage
    -----
    >>> scorer = CustomerSentimentScorer()
    >>> scorer.load()
    >>> results = scorer.score(["Great experience!", "Terrible service."])
    >>> df_out = scorer.score_df(df, text_col="review_text")
    """

    def __init__(
        self,
        checkpoint: str = MODEL_CHECKPOINT,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: int = -1,  # -1 = CPU, 0 = first GPU
    ) -> None:
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.device = device
        self._pipeline: Any = None
        self.metrics_: dict[str, float] = {}

    # -- public API -----------------------------------------------------------
    def load(self) -> CustomerSentimentScorer:
        """Download / load the model. Call once before scoring."""
        from transformers import pipeline  # lazy import — heavy dependency

        logger.info("Loading sentiment model: %s", self.checkpoint)
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.checkpoint,
            tokenizer=self.checkpoint,
            device=self.device,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        logger.info("Sentiment model loaded.")
        return self

    def score(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Score a list of review strings.

        Returns
        -------
        list of dicts, each with:
          - label   : "POSITIVE" or "NEGATIVE"
          - score   : model confidence (0–1)
          - numeric : 1.0 for POSITIVE, 0.0 for NEGATIVE
        """
        self._assert_loaded()
        results = self._batch_infer(texts)
        return results

    def score_df(
        self,
        df: pd.DataFrame,
        text_col: str = "review_text",
        prefix: str = "sentiment",
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a DataFrame in-place (copy).

        New columns:
          {prefix}_label   — POSITIVE / NEGATIVE
          {prefix}_score   — model confidence
          {prefix}_numeric — 1 / 0
        """
        self._assert_loaded()
        df = df.copy()
        texts = df[text_col].fillna("").tolist()
        results = self._batch_infer(texts)

        df[f"{prefix}_label"] = [r["label"] for r in results]
        df[f"{prefix}_score"] = [r["score"] for r in results]
        df[f"{prefix}_numeric"] = [r["numeric"] for r in results]
        return df

    def evaluate(
        self,
        texts: list[str],
        true_labels: list[int],  # 1 = POSITIVE, 0 = NEGATIVE
    ) -> dict[str, float]:
        """
        Compute classification metrics against ground-truth binary labels.
        Stores results in self.metrics_.
        """
        self._assert_loaded()
        results = self._batch_infer(texts)
        predicted = [r["numeric"] for r in results]

        report = classification_report(
            true_labels, predicted, target_names=["NEGATIVE", "POSITIVE"], output_dict=True, zero_division=0
        )
        self.metrics_ = {
            "accuracy": float(report["accuracy"]),
            "f1_positive": float(report["POSITIVE"]["f1-score"]),
            "precision_positive": float(report["POSITIVE"]["precision"]),
            "recall_positive": float(report["POSITIVE"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
        }
        logger.info("Sentiment evaluation metrics: %s", self.metrics_)
        return self.metrics_

    def log_to_mlflow(self) -> str:
        """Log model config and metrics to MLflow."""
        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name="distilbert_sentiment") as run:
            mlflow.log_params(
                {
                    "checkpoint": self.checkpoint,
                    "batch_size": self.batch_size,
                    "max_length": MAX_LENGTH,
                }
            )
            if self.metrics_:
                mlflow.log_metrics(self.metrics_)
            logger.info("Logged sentiment model to MLflow run %s", run.info.run_id)
            return run.info.run_id

    # -- internal helpers -----------------------------------------------------
    def _batch_infer(self, texts: list[str]) -> list[dict[str, Any]]:
        """Run inference in batches, return normalised result dicts."""
        results: list[dict[str, Any]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            # Replace empty strings so the model doesn't error
            chunk = [t if t.strip() else "no review provided" for t in chunk]
            raw = self._pipeline(chunk)
            for item in raw:
                label = item["label"].upper()
                confidence = float(item["score"])
                results.append(
                    {
                        "label": label,
                        "score": confidence,
                        "numeric": 1 if label == "POSITIVE" else 0,
                    }
                )
        return results

    def _assert_loaded(self) -> None:
        if self._pipeline is None:
            raise RuntimeError("Model not loaded. Call .load() before scoring.")


# ── Convenience factory ───────────────────────────────────────────────────────
def build_scorer(device: int = -1) -> CustomerSentimentScorer:
    """Create and load a ready-to-use scorer in one call."""
    return CustomerSentimentScorer(device=device).load()
