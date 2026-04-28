#!/usr/bin/env python
"""
ML Verification Suite — Day 4
================================
Runs all three models on synthetic data and asserts minimum quality gates.

Targets (must all pass for CI green):
  Aging Model    : RMSE < 30, R² > 0.50
  Rep Clustering : Silhouette score > 0.30
  Sentiment      : F1 (macro) > 0.70  (DistilBERT pre-trained, no fine-tuning)

Usage
-----
    python scripts/verify_ml.py
    python scripts/verify_ml.py --skip-sentiment   # skip heavy model load
    python scripts/verify_ml.py --verbose
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dealership.ml.aging_model import InventoryAgingModel
from dealership.ml.rep_clustering import RepClusteringModel
from dealership.ml.sentiment import CustomerSentimentScorer

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
AGING_MAX_RMSE = 30.0
AGING_MIN_R2 = 0.50
CLUSTERING_MIN_SILHOUETTE = 0.30
SENTIMENT_MIN_F1 = 0.70


# ── Synthetic data generators ─────────────────────────────────────────────────
def _make_inventory_df(n: int = 800, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    makes = ["Toyota", "Ford", "Honda", "Chevrolet", "BMW", "Hyundai"]
    models = ["Sedan", "SUV", "Truck", "Coupe", "Hatchback"]
    colors = ["White", "Black", "Silver", "Red", "Blue"]
    fuels = ["Gasoline", "Hybrid", "Electric", "Diesel"]
    transmissions = ["Automatic", "Manual", "CVT"]
    trims = ["Base", "Sport", "Premium", "Limited"]

    make = rng.choice(makes, n)
    year = rng.integers(2015, 2025, n)
    mileage = rng.integers(0, 120_000, n)
    price = rng.integers(15_000, 65_000, n)
    cpo = rng.choice([True, False], n)
    price_drop_days = rng.integers(0, 60, n)

    # Target: roughly correlated with age + mileage + price
    vehicle_age = 2025 - year
    days_on_lot = (
        (10 + vehicle_age * 3 + mileage / 5_000 - price / 5_000 + rng.normal(0, 8, n)).clip(1, 180).astype(int)
    )

    return pd.DataFrame(
        {
            "make": make,
            "model": rng.choice(models, n),
            "trim": rng.choice(trims, n),
            "color": rng.choice(colors, n),
            "fuel_type": rng.choice(fuels, n),
            "transmission": rng.choice(transmissions, n),
            "year": year,
            "mileage": mileage,
            "price": price,
            "certified_pre_owned": cpo,
            "days_since_last_price_drop": price_drop_days,
            "days_on_lot": days_on_lot,
        }
    )


def _make_rep_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Create 4 natural clusters manually for clean silhouette
    archetypes = {
        "Closer": dict(
            close_rate=(0.55, 0.10),
            avg_deal_value=(42_000, 5_000),
            deals_closed=(18, 4),
            avg_days_to_close=(8, 2),
            follow_up_rate=(0.60, 0.10),
            customer_satisfaction=(3.9, 0.3),
            upsell_rate=(0.35, 0.08),
        ),
        "Volume Player": dict(
            close_rate=(0.35, 0.08),
            avg_deal_value=(28_000, 4_000),
            deals_closed=(35, 5),
            avg_days_to_close=(6, 2),
            follow_up_rate=(0.45, 0.10),
            customer_satisfaction=(3.6, 0.4),
            upsell_rate=(0.20, 0.06),
        ),
        "Nurturer": dict(
            close_rate=(0.40, 0.08),
            avg_deal_value=(32_000, 5_000),
            deals_closed=(15, 3),
            avg_days_to_close=(14, 3),
            follow_up_rate=(0.85, 0.08),
            customer_satisfaction=(4.6, 0.2),
            upsell_rate=(0.45, 0.08),
        ),
        "Struggler": dict(
            close_rate=(0.15, 0.06),
            avg_deal_value=(22_000, 4_000),
            deals_closed=(8, 3),
            avg_days_to_close=(22, 5),
            follow_up_rate=(0.30, 0.10),
            customer_satisfaction=(3.0, 0.5),
            upsell_rate=(0.10, 0.05),
        ),
    }
    rows = []
    per_archetype = n // len(archetypes)
    for arch, params in archetypes.items():
        for _ in range(per_archetype):
            row = {"archetype_true": arch}
            for feat, (mean, std) in params.items():
                row[feat] = float(np.clip(rng.normal(mean, std), 0, None))
            rows.append(row)
    return pd.DataFrame(rows).drop(columns=["archetype_true"])


def _make_review_data(n: int = 100) -> tuple[list[str], list[int]]:
    positives = [
        "Absolutely loved the experience, staff was so helpful.",
        "Great service, will definitely come back!",
        "The team was professional and the car was exactly what I needed.",
        "Smooth process from start to finish. Highly recommend.",
        "Fantastic dealership, very transparent pricing.",
        "My salesperson was knowledgeable and not pushy at all.",
        "I got a great deal and the financing was straightforward.",
        "Best car buying experience I've had in years.",
    ]
    negatives = [
        "Terrible experience, they were dishonest about the fees.",
        "Waited three hours and nothing was sorted. Awful.",
        "The car had issues they didn't disclose. Deeply disappointed.",
        "Very pushy salespeople, I felt pressured the entire time.",
        "Finance process was a nightmare, hidden charges everywhere.",
        "Would not recommend. Staff was rude and unhelpful.",
        "They promised a callback and never followed up. Unacceptable.",
        "The vehicle I reserved was sold to someone else. Furious.",
    ]
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for _ in range(n):
        if rng.random() > 0.45:
            texts.append(rng.choice(positives))
            labels.append(1)
        else:
            texts.append(rng.choice(negatives))
            labels.append(0)
    return texts, labels


# ── Check helpers ─────────────────────────────────────────────────────────────
_PASS = "✅ PASS"
_FAIL = "❌ FAIL"


def _check(label: str, value: float, threshold: float, mode: str = "min") -> bool:
    ok = value >= threshold if mode == "min" else value <= threshold
    symbol = _PASS if ok else _FAIL
    comp = ">=" if mode == "min" else "<="
    print(f"  {symbol}  {label}: {value:.4f} {comp} {threshold}")
    return ok


# ── Verification routines ─────────────────────────────────────────────────────
def verify_aging(verbose: bool = False) -> bool:
    print("\n📦  [1/3] Inventory Aging Model (XGBoost)")
    print("  " + "─" * 48)
    df = _make_inventory_df()
    model = InventoryAgingModel()
    model.fit(df)

    passed = True
    passed &= _check("RMSE", model.metrics_["rmse"], AGING_MAX_RMSE, mode="max")
    passed &= _check("R²", model.metrics_["r2"], AGING_MIN_R2, mode="min")

    if verbose:
        expl = model.explain(df.head(3).drop(columns=["days_on_lot"]))
        print("\n  Top-5 SHAP feature importances:")
        for feat, imp in list(expl["feature_importance"].items())[:5]:
            print(f"    {feat:<28} {imp:.4f}")
    return passed


def verify_clustering(verbose: bool = False) -> bool:
    print("\n👥  [2/3] Rep Clustering (K-Means archetypes)")
    print("  " + "─" * 48)
    df = _make_rep_df()
    model = RepClusteringModel()
    model.fit(df)

    passed = _check(
        "Silhouette score",
        model.metrics_["silhouette_score"],
        CLUSTERING_MIN_SILHOUETTE,
    )
    print(f"  ℹ️   k={int(model.metrics_['n_clusters'])}  Archetypes: {model.archetype_labels_}")

    if verbose:
        print("\n  Archetype centroid summary:")
        print(model.archetype_summary().to_string(float_format=lambda x: f"{x:.3f}"))
    return passed


def verify_sentiment(verbose: bool = False) -> bool:
    print("\n💬  [3/3] Sentiment Scorer (DistilBERT)")
    print("  " + "─" * 48)
    texts, labels = _make_review_data(100)
    scorer = CustomerSentimentScorer()
    scorer.load()
    scorer.evaluate(texts, labels)

    passed = _check(
        "F1 macro",
        scorer.metrics_["f1_macro"],
        SENTIMENT_MIN_F1,
    )
    if verbose:
        print(f"  Accuracy : {scorer.metrics_['accuracy']:.3f}")
        print(f"  Precision: {scorer.metrics_['precision_positive']:.3f}")
        print(f"  Recall   : {scorer.metrics_['recall_positive']:.3f}")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify Day 4 ML models.")
    p.add_argument("--skip-sentiment", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )
    for lib in ("transformers", "xgboost", "mlflow", "urllib3"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    print("\n" + "═" * 55)
    print("  Dealership Intelligence Agent — Day 4 ML Verification")
    print("═" * 55)

    results: list[bool] = []
    results.append(verify_aging(args.verbose))
    results.append(verify_clustering(args.verbose))

    if not args.skip_sentiment:
        results.append(verify_sentiment(args.verbose))
    else:
        print("\n💬  [3/3] Sentiment Scorer — SKIPPED (--skip-sentiment)")

    total = len(results)
    passed = sum(results)

    print("\n" + "═" * 55)
    print(f"  Result: {passed}/{total} checks passed")
    if passed == total:
        print("  🎉  Day 4 ML layer verified successfully!")
    else:
        print("  ⚠️   Some checks failed — review output above.")
    print("═" * 55 + "\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
