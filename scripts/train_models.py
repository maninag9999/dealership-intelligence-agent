#!/usr/bin/env python
"""
CLI entrypoint for the Dealership ML training pipeline.

Usage examples
--------------
# Full training run (default DB path)
python scripts/train_models.py

# Point at a specific DB
python scripts/train_models.py --db data/dealership.duckdb

# Skip slow DistilBERT eval for quick iterations
python scripts/train_models.py --skip-sentiment

# Combine flags
python scripts/train_models.py --db data/dealership.duckdb --skip-sentiment --log-level DEBUG
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root without `pip install -e .`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dealership.ml.train import TrainingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train all Dealership Intelligence ML models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to DuckDB file. Defaults to DEALERSHIP_DB_PATH env var or data/dealership.duckdb",
    )
    parser.add_argument(
        "--skip-sentiment",
        action="store_true",
        help="Skip DistilBERT sentiment evaluation (faster for dev iterations).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        level=getattr(logging, level),
    )
    # Quieten noisy third-party loggers
    for noisy in ("transformers", "xgboost", "mlflow", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    kwargs: dict = {"skip_sentiment": args.skip_sentiment}
    if args.db is not None:
        kwargs["db_path"] = args.db

    pipeline = TrainingPipeline(**kwargs)
    results = pipeline.run()

    # Pretty-print summary
    print("\n" + "─" * 55)
    print("  TRAINING SUMMARY")
    print("─" * 55)

    if results.aging_metrics:
        m = results.aging_metrics
        print(
            f"  📦  Aging Model   RMSE={m.get('rmse', 'N/A'):.2f}  MAE={m.get('mae', 'N/A'):.2f}  R²={m.get('r2', 'N/A'):.3f}"
        )

    if results.clustering_metrics:
        m = results.clustering_metrics
        print(f"  👥  Rep Clusters  k={int(m.get('n_clusters', 0))}  Silhouette={m.get('silhouette_score', 'N/A'):.4f}")

    if results.sentiment_metrics:
        m = results.sentiment_metrics
        print(f"  💬  Sentiment     Accuracy={m.get('accuracy', 'N/A'):.3f}  F1(macro)={m.get('f1_macro', 'N/A'):.3f}")

    print("─" * 55)
    print(f"  MLflow run IDs: {results.run_ids}")

    if results.errors:
        print("\n  ❌  Errors encountered:")
        for err in results.errors:
            print(f"      • {err}")
        print("─" * 55)
        return 1

    print("\n  ✅  All models trained and logged to MLflow.")
    print("─" * 55)
    return 0


if __name__ == "__main__":
    sys.exit(main())
