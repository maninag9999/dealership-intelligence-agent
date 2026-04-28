"""
Sales Rep Clustering
--------------------
K-Means clustering that groups sales reps into behavioural archetypes
based on their aggregated performance features.

Archetypes (auto-assigned after fitting via centroid heuristics):
  - "Closer"       — high close_rate, high avg_deal_value
  - "Volume Player" — high deals_closed, moderate margins
  - "Nurturer"     — high follow_up_rate, high customer_satisfaction
  - "Struggler"    — below-average across KPIs

MLflow tracking is included for the clustering run.
"""

from __future__ import annotations

import logging

import mlflow
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_NAME = "rep_clustering_kmeans"
EXPERIMENT_NAME = "dealership/rep_clustering"

# Features that describe a rep's aggregated behaviour
REP_FEATURE_COLS = [
    "close_rate",  # fraction of leads converted
    "avg_deal_value",  # average gross per deal
    "deals_closed",  # total deals in period
    "avg_days_to_close",  # speed metric (lower = faster)
    "follow_up_rate",  # fraction of leads with ≥1 follow-up
    "customer_satisfaction",  # avg CSAT score (1-5)
    "upsell_rate",  # fraction of deals with add-ons
]

# Human-readable archetype names keyed by a tuple of dominant-feature indices
# Order of REP_FEATURE_COLS matters here for centroid comparison
_ARCHETYPE_MAP = {
    "Closer": {"close_rate": "high", "avg_deal_value": "high"},
    "Volume Player": {"deals_closed": "high", "avg_days_to_close": "low"},
    "Nurturer": {"follow_up_rate": "high", "customer_satisfaction": "high"},
    "Struggler": {},  # fallback
}

MAX_K = 8  # Maximum clusters to evaluate during elbow search


# ── Clustering model ─────────────────────────────────────────────────────────
class RepClusteringModel:
    """
    K-Means rep archetype clustering.

    Usage
    -----
    >>> model = RepClusteringModel()
    >>> model.fit(df_reps)          # df_reps has rep-level aggregated KPIs
    >>> labels = model.predict(df_new_reps)
    >>> print(model.archetype_summary())
    """

    def __init__(self, n_clusters: int | None = None) -> None:
        self.n_clusters = n_clusters  # None = auto-select via elbow
        self.scaler = StandardScaler()
        self.kmeans: KMeans | None = None
        self.archetype_labels_: list[str] = []
        self.metrics_: dict[str, float] = {}
        self._cluster_centers_df: pd.DataFrame | None = None

    # -- public API -----------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> RepClusteringModel:
        """Fit on a rep-level aggregated DataFrame."""
        X = self._prepare(df, fit=True)

        k = self.n_clusters or self._select_k(X)
        logger.info("Using k=%d clusters for rep archetypes", k)

        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(X)

        sil = float(silhouette_score(X, labels))
        self.metrics_ = {"silhouette_score": sil, "n_clusters": float(k)}
        logger.info("Rep clustering metrics: %s", self.metrics_)

        # Store centroid positions in original feature space for interpretation
        centers_scaled = self.kmeans.cluster_centers_
        centers_original = self.scaler.inverse_transform(centers_scaled)
        self._cluster_centers_df = pd.DataFrame(centers_original, columns=self._feature_cols)

        # Assign archetype names to each cluster
        self.archetype_labels_ = self._assign_archetypes(self._cluster_centers_df)
        logger.info("Archetype assignments: %s", dict(enumerate(self.archetype_labels_)))

        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Return archetype name (string) for each row."""
        self._assert_fitted()
        X = self._prepare(df, fit=False)
        cluster_ids = self.kmeans.predict(X)  # type: ignore[union-attr]
        return pd.Series(
            [self.archetype_labels_[c] for c in cluster_ids],
            index=df.index,
            name="archetype",
        )

    def archetype_summary(self) -> pd.DataFrame:
        """Return mean KPI values per archetype for reporting."""
        self._assert_fitted()
        df = self._cluster_centers_df.copy()  # type: ignore[union-attr]
        df.insert(0, "archetype", self.archetype_labels_)
        return df.set_index("archetype").round(3)

    def log_to_mlflow(self) -> str:
        """Log params, metrics, and cluster summary to MLflow."""
        self._assert_fitted()
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=MODEL_NAME) as run:
            mlflow.log_params({"n_clusters": self.metrics_["n_clusters"]})
            mlflow.log_metrics({"silhouette_score": self.metrics_["silhouette_score"]})

            # Log archetype summary as a CSV artifact
            summary = self.archetype_summary()
            summary_path = "/tmp/rep_archetypes.csv"
            summary.to_csv(summary_path)
            mlflow.log_artifact(summary_path, artifact_path="clustering")

            logger.info("Logged rep clustering to MLflow run %s", run.info.run_id)
            return run.info.run_id

    # -- internal helpers -----------------------------------------------------
    @property
    def _feature_cols(self) -> list[str]:
        return [c for c in REP_FEATURE_COLS if c in self._fitted_cols]

    def _prepare(self, df: pd.DataFrame, *, fit: bool) -> np.ndarray:
        available = [c for c in REP_FEATURE_COLS if c in df.columns]
        if fit:
            self._fitted_cols = available
        X = df[available].fillna(df[available].median())
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)

    def _select_k(self, X: np.ndarray) -> int:
        """Elbow method: pick k that maximises silhouette score."""
        max_k = min(MAX_K, len(X) - 1)
        if max_k < 2:
            return 2

        best_k, best_sil = 2, -1.0
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels)
            logger.debug("k=%d → silhouette=%.4f", k, sil)
            if sil > best_sil:
                best_sil, best_k = sil, k

        return best_k

    def _assign_archetypes(self, centers: pd.DataFrame) -> list[str]:
        """
        Map each cluster index to a human-readable archetype name.

        Strategy:
          1. Compute per-feature percentile rank across clusters.
          2. Score each cluster against each archetype's signature.
          3. Greedy assignment: best-matching cluster gets the archetype;
             remainder fall back to "Contributor N".
        """
        n = len(centers)
        ranks = centers.rank(pct=True)  # 0–1, higher = better rank

        # "avg_days_to_close" is inverted (lower = better closer)
        if "avg_days_to_close" in ranks.columns:
            ranks["avg_days_to_close"] = 1.0 - ranks["avg_days_to_close"]

        archetype_order = ["Closer", "Volume Player", "Nurturer", "Struggler"]
        signatures: dict[str, dict[str, str]] = {
            "Closer": {"close_rate": "high", "avg_deal_value": "high"},
            "Volume Player": {"deals_closed": "high", "avg_days_to_close": "low"},
            "Nurturer": {"follow_up_rate": "high", "customer_satisfaction": "high"},
            "Struggler": {},
        }

        # Score matrix: clusters × archetypes
        score_matrix = np.zeros((n, len(archetype_order)))
        for j, arch in enumerate(archetype_order):
            for feat, direction in signatures[arch].items():
                if feat in ranks.columns:
                    col_ranks = ranks[feat].values
                    if direction == "high":
                        score_matrix[:, j] += col_ranks
                    else:
                        score_matrix[:, j] += 1.0 - col_ranks

        # Greedy assignment (no archetype used twice)
        assigned: dict[int, str] = {}
        available_archetypes = list(archetype_order)

        for _ in range(min(n, len(archetype_order))):
            if not available_archetypes:
                break
            # Find best unassigned (cluster, archetype) pair
            best_score = -1.0
            best_cluster = best_arch = None
            for i in range(n):
                if i in assigned:
                    continue
                for j, arch in enumerate(archetype_order):
                    if arch not in available_archetypes:
                        continue
                    if score_matrix[i, j] > best_score:
                        best_score = score_matrix[i, j]
                        best_cluster, best_arch = i, arch
            if best_cluster is not None and best_arch is not None:
                assigned[best_cluster] = best_arch
                available_archetypes.remove(best_arch)

        # Fill remaining clusters
        contributor_idx = 1
        result = []
        for i in range(n):
            if i in assigned:
                result.append(assigned[i])
            else:
                result.append(f"Contributor {contributor_idx}")
                contributor_idx += 1
        return result

    def _assert_fitted(self) -> None:
        if self.kmeans is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
