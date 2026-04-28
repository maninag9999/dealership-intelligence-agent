"""
Agent Tools
-----------
LangChain-compatible tools that the LangGraph agent can invoke.

Each tool is a plain Python function decorated with @tool.
The agent decides which tools to call based on the user's question.

Tools:
  1. search_inventory        — ChromaDB semantic search
  2. predict_days_on_lot     — XGBoost aging model
  3. get_rep_archetypes      — K-Means clustering
  4. score_customer_sentiment— DistilBERT sentiment
  5. query_inventory_stats   — DuckDB live stats
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = Path(os.getenv("DEALERSHIP_DB_PATH", "data/dealership.duckdb"))
CHROMA_PATH = os.getenv("CHROMA_PATH", "data/chroma_db")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "vehicle_inventory")


# ── Lazy singletons — models loaded once on first call ────────────────────────
_aging_model = None
_clustering_model = None
_sentiment_scorer = None
_chroma_collection = None


def _get_aging_model():
    global _aging_model
    if _aging_model is None:
        logger.info("Loading aging model…")
        # Train on-the-fly with a small synthetic sample for demo
        # In production: load from MLflow model registry
        _aging_model = _build_demo_aging_model()
    return _aging_model


def _get_clustering_model():
    global _clustering_model
    if _clustering_model is None:
        logger.info("Loading clustering model…")
        _clustering_model = _build_demo_clustering_model()
    return _clustering_model


def _get_sentiment_scorer():
    global _sentiment_scorer
    if _sentiment_scorer is None:
        from dealership.ml.sentiment import CustomerSentimentScorer

        logger.info("Loading sentiment scorer…")
        _sentiment_scorer = CustomerSentimentScorer()
        _sentiment_scorer.load()
    return _sentiment_scorer


def _get_chroma_collection():
    global _chroma_collection
    if _chroma_collection is None:
        import chromadb

        logger.info("Connecting to ChromaDB at %s…", CHROMA_PATH)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        try:
            _chroma_collection = client.get_collection(CHROMA_COLLECTION)
        except Exception:
            # Collection doesn't exist yet — create a minimal demo one
            _chroma_collection = _build_demo_chroma_collection(client)
    return _chroma_collection


# ── Tool 1: Inventory Search ──────────────────────────────────────────────────
@tool
def search_inventory(query: str, n_results: int = 5) -> str:
    """
    Search vehicle inventory using natural language.
    Returns the top matching vehicles with their key attributes.

    Args:
        query: Natural language search query (e.g. 'red SUV under 30000')
        n_results: Number of results to return (default 5)
    """
    try:
        collection = _get_chroma_collection()
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count()),
        )

        if not results["documents"] or not results["documents"][0]:
            return "No matching vehicles found in inventory."

        output = [f"Found {len(results['documents'][0])} matching vehicles:\n"]
        for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0], strict=False), 1):
            output.append(f"{i}. {doc}")
            if meta:
                days = meta.get("days_on_lot", "N/A")
                price = meta.get("price", "N/A")
                output.append(f"   Price: ${price:,} | Days on lot: {days}")

        return "\n".join(output)

    except Exception as exc:
        logger.exception("search_inventory failed")
        return f"Inventory search failed: {exc}"


# ── Tool 2: Days-on-Lot Prediction ───────────────────────────────────────────
@tool
def predict_days_on_lot(vehicle_data: str) -> str:
    """
    Predict how many days a vehicle will remain on the lot.
    Provide vehicle attributes as a JSON string.

    Args:
        vehicle_data: JSON string with keys: make, model, year, mileage,
                      price, trim, color, fuel_type, transmission,
                      certified_pre_owned (bool), days_since_last_price_drop
    Example:
        '{"make": "Toyota", "model": "SUV", "year": 2022, "mileage": 15000,
          "price": 32000, "trim": "Sport", "color": "White",
          "fuel_type": "Gasoline", "transmission": "Automatic",
          "certified_pre_owned": true, "days_since_last_price_drop": 0}'
    """
    try:
        data = json.loads(vehicle_data)
        df = pd.DataFrame([data])

        model = _get_aging_model()
        prediction = float(model.predict(df)[0])
        explanation = model.explain(df)

        top_factors = list(explanation["feature_importance"].items())[:5]

        output = [
            f"Predicted days on lot: {prediction:.0f} days",
            "",
            "Top factors driving this prediction:",
        ]
        for feat, importance in top_factors:
            output.append(f"  • {feat}: {importance:.3f} impact")

        # Risk assessment
        if prediction > 60:
            output.append("\n⚠️  HIGH RISK: Consider a price reduction or promotion.")
        elif prediction > 30:
            output.append("\n⚡ MODERATE RISK: Monitor closely over the next 2 weeks.")
        else:
            output.append("\n✅ LOW RISK: This vehicle should move quickly.")

        return "\n".join(output)

    except json.JSONDecodeError:
        return "Invalid JSON format. Please provide vehicle data as a valid JSON string."
    except Exception as exc:
        logger.exception("predict_days_on_lot failed")
        return f"Prediction failed: {exc}"


# ── Tool 3: Rep Archetypes ────────────────────────────────────────────────────
@tool
def get_rep_archetypes(rep_data: str = "") -> str:
    """
    Get sales rep performance archetypes and KPI analysis.
    Optionally provide specific rep data as JSON for individual analysis.

    Args:
        rep_data: Optional JSON string with rep KPIs for individual analysis.
                  If empty, returns the archetype summary for the whole team.
    Example:
        '{"close_rate": 0.45, "avg_deal_value": 35000, "deals_closed": 22,
          "avg_days_to_close": 10, "follow_up_rate": 0.7,
          "customer_satisfaction": 4.2, "upsell_rate": 0.3}'
    """
    try:
        model = _get_clustering_model()
        summary = model.archetype_summary()

        output = ["Sales Rep Archetype Analysis\n" + "=" * 40]

        # Show archetype profiles
        output.append("\nTeam Archetype Profiles:")
        for archetype, row in summary.iterrows():
            output.append(f"\n🏷️  {archetype}")
            output.append(f"   Close Rate:      {row.get('close_rate', 0):.1%}")
            output.append(f"   Avg Deal Value:  ${row.get('avg_deal_value', 0):,.0f}")
            output.append(f"   Deals Closed:    {row.get('deals_closed', 0):.0f}")
            output.append(f"   Days to Close:   {row.get('avg_days_to_close', 0):.0f}")
            output.append(f"   CSAT Score:      {row.get('customer_satisfaction', 0):.1f}/5.0")

        # Coaching recommendations per archetype
        coaching = {
            "Closer": "✅ Top performer — mentor other reps on closing techniques.",
            "Volume Player": "📈 High volume but lower margins — focus on upsell training.",
            "Nurturer": "💛 Excellent CSAT — coach on urgency and closing speed.",
            "Struggler": "🚨 Needs immediate coaching plan and pipeline review.",
        }
        output.append("\n\nCoaching Recommendations:")
        for arch, advice in coaching.items():
            if arch in summary.index:
                output.append(f"  {arch}: {advice}")

        # Individual rep prediction if data provided
        if rep_data.strip():
            data = json.loads(rep_data)
            df = pd.DataFrame([data])
            archetype = model.predict(df).iloc[0]
            output.append(f"\n\n📊 Individual Rep Assessment: {archetype}")
            output.append(f"   {coaching.get(archetype, 'Continue current approach.')}")

        return "\n".join(output)

    except Exception as exc:
        logger.exception("get_rep_archetypes failed")
        return f"Rep archetype analysis failed: {exc}"


# ── Tool 4: Customer Sentiment ────────────────────────────────────────────────
@tool
def score_customer_sentiment(reviews: str) -> str:
    """
    Analyse sentiment of customer reviews.
    Provide reviews as a JSON array of strings or a single review string.

    Args:
        reviews: Either a single review string or JSON array of review strings.
    Example (single):  "Great experience, very helpful staff!"
    Example (batch):   '["Amazing!", "Terrible service.", "It was okay."]'
    """
    try:
        # Parse input — accept both single string and JSON array
        review_list = json.loads(reviews) if reviews.strip().startswith("[") else [reviews]

        scorer = _get_sentiment_scorer()
        results = scorer.score(review_list)

        positive = sum(1 for r in results if r["label"] == "POSITIVE")
        negative = len(results) - positive
        avg_confidence = np.mean([r["score"] for r in results])

        output = [
            f"Sentiment Analysis — {len(results)} review(s)",
            "=" * 40,
            f"✅ Positive: {positive} ({positive/len(results):.0%})",
            f"❌ Negative: {negative} ({negative/len(results):.0%})",
            f"📊 Avg Confidence: {avg_confidence:.2%}",
        ]

        if len(review_list) <= 5:
            output.append("\nIndividual Results:")
            for i, (review, result) in enumerate(zip(review_list, results, strict=False), 1):
                emoji = "✅" if result["label"] == "POSITIVE" else "❌"
                truncated = review[:60] + "…" if len(review) > 60 else review
                output.append(f"  {i}. {emoji} {result['label']} ({result['score']:.0%}) — \"{truncated}\"")

        # Overall recommendation
        positive_rate = positive / len(results)
        if positive_rate >= 0.8:
            output.append("\n🌟 Excellent customer sentiment — share these wins with the team!")
        elif positive_rate >= 0.6:
            output.append("\n⚡ Mixed sentiment — identify and address recurring complaints.")
        else:
            output.append("\n🚨 Poor sentiment — urgent review of customer experience needed.")

        return "\n".join(output)

    except Exception as exc:
        logger.exception("score_customer_sentiment failed")
        return f"Sentiment scoring failed: {exc}"


# ── Tool 5: Inventory Stats ───────────────────────────────────────────────────
@tool
def query_inventory_stats(metric: str = "overview") -> str:
    """
    Query live inventory statistics from the database.

    Args:
        metric: One of 'overview', 'aging', 'pricing', 'makes'
                - overview : total count, avg days-on-lot, avg price
                - aging    : vehicles by aging bucket (0-30, 31-60, 60+ days)
                - pricing  : price distribution by make
                - makes    : inventory count by make
    """
    if not DB_PATH.exists():
        return _demo_inventory_stats(metric)

    try:
        with duckdb.connect(str(DB_PATH), read_only=True) as con:
            if metric == "aging":
                df = con.execute("""
                    SELECT
                        CASE
                            WHEN days_on_lot <= 30 THEN '0-30 days (Fresh)'
                            WHEN days_on_lot <= 60 THEN '31-60 days (Watch)'
                            ELSE '60+ days (At Risk)'
                        END AS bucket,
                        COUNT(*) AS vehicles,
                        ROUND(AVG(price), 0) AS avg_price
                    FROM mart_inventory
                    GROUP BY 1
                    ORDER BY 1
                """).df()
                return _format_df_output("Inventory Aging Buckets", df)

            elif metric == "pricing":
                df = con.execute("""
                    SELECT make,
                           COUNT(*) AS vehicles,
                           ROUND(AVG(price), 0) AS avg_price,
                           MIN(price) AS min_price,
                           MAX(price) AS max_price
                    FROM mart_inventory
                    GROUP BY make
                    ORDER BY avg_price DESC
                    LIMIT 10
                """).df()
                return _format_df_output("Pricing by Make (Top 10)", df)

            elif metric == "makes":
                df = con.execute("""
                    SELECT make, COUNT(*) AS count,
                           ROUND(AVG(days_on_lot), 1) AS avg_days
                    FROM mart_inventory
                    GROUP BY make
                    ORDER BY count DESC
                """).df()
                return _format_df_output("Inventory by Make", df)

            else:  # overview
                row = con.execute("""
                    SELECT
                        COUNT(*)                    AS total_vehicles,
                        ROUND(AVG(days_on_lot), 1)  AS avg_days_on_lot,
                        ROUND(AVG(price), 0)         AS avg_price,
                        MIN(price)                   AS min_price,
                        MAX(price)                   AS max_price,
                        SUM(CASE WHEN days_on_lot > 60 THEN 1 ELSE 0 END) AS at_risk
                    FROM mart_inventory
                """).fetchone()
                return (
                    f"Inventory Overview\n{'='*40}\n"
                    f"Total Vehicles:    {row[0]:,}\n"
                    f"Avg Days on Lot:   {row[1]} days\n"
                    f"Avg Price:         ${row[2]:,}\n"
                    f"Price Range:       ${row[3]:,} - ${row[4]:,}\n"
                    f"At Risk (60+ days): {row[5]} vehicles"
                )

    except Exception as exc:
        logger.exception("query_inventory_stats failed")
        return f"Stats query failed: {exc}"


# ── Demo data builders (used when DB/ChromaDB not available) ──────────────────
def _build_demo_aging_model():
    """Build and return a trained aging model on synthetic data."""
    import numpy as np

    from dealership.ml.aging_model import InventoryAgingModel

    rng = np.random.default_rng(42)
    n = 500
    makes = ["Toyota", "Ford", "Honda", "Chevrolet", "BMW"]
    df = pd.DataFrame(
        {
            "make": rng.choice(makes, n),
            "model": rng.choice(["Sedan", "SUV", "Truck"], n),
            "trim": rng.choice(["Base", "Sport", "Premium"], n),
            "color": rng.choice(["White", "Black", "Silver"], n),
            "fuel_type": rng.choice(["Gasoline", "Hybrid", "Electric"], n),
            "transmission": rng.choice(["Automatic", "Manual"], n),
            "year": rng.integers(2016, 2025, n),
            "mileage": rng.integers(0, 100_000, n),
            "price": rng.integers(15_000, 60_000, n),
            "certified_pre_owned": rng.choice([True, False], n),
            "days_since_last_price_drop": rng.integers(0, 45, n),
            "days_on_lot": (10 + rng.normal(20, 10, n)).clip(1, 120).astype(int),
        }
    )
    model = InventoryAgingModel()
    model.fit(df)
    return model


def _build_demo_clustering_model():
    """Build and return a trained clustering model on synthetic data."""
    import numpy as np

    from dealership.ml.rep_clustering import RepClusteringModel

    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame(
        {
            "close_rate": rng.uniform(0.1, 0.7, n),
            "avg_deal_value": rng.uniform(18_000, 50_000, n),
            "deals_closed": rng.integers(5, 40, n),
            "avg_days_to_close": rng.uniform(4, 25, n),
            "follow_up_rate": rng.uniform(0.2, 0.95, n),
            "customer_satisfaction": rng.uniform(2.5, 5.0, n),
            "upsell_rate": rng.uniform(0.05, 0.55, n),
        }
    )
    model = RepClusteringModel()
    model.fit(df)
    return model


def _build_demo_chroma_collection(client):
    """Create a small demo ChromaDB collection with sample vehicles."""

    collection = client.get_or_create_collection(CHROMA_COLLECTION)
    if collection.count() > 0:
        return collection

    vehicles = [
        (
            "2023 Toyota RAV4 Sport White SUV Gasoline",
            {"make": "Toyota", "model": "RAV4", "price": 34500, "days_on_lot": 12},
        ),
        (
            "2022 Ford F-150 XLT Silver Truck Gasoline",
            {"make": "Ford", "model": "F-150", "price": 42000, "days_on_lot": 28},
        ),
        ("2023 Honda CR-V EX Blue SUV Hybrid", {"make": "Honda", "model": "CR-V", "price": 31000, "days_on_lot": 8}),
        (
            "2021 BMW 3 Series 330i Black Sedan Gasoline",
            {"make": "BMW", "model": "3 Series", "price": 38000, "days_on_lot": 45},
        ),
        (
            "2022 Chevrolet Silverado LT Red Truck Gasoline",
            {"make": "Chevrolet", "model": "Silverado", "price": 39500, "days_on_lot": 19},
        ),
        (
            "2023 Hyundai Tucson SEL White SUV Gasoline",
            {"make": "Hyundai", "model": "Tucson", "price": 28000, "days_on_lot": 6},
        ),
        (
            "2022 Tesla Model 3 Long Range Blue Sedan Electric",
            {"make": "Tesla", "model": "Model 3", "price": 45000, "days_on_lot": 3},
        ),
        (
            "2021 Toyota Camry SE Gray Sedan Gasoline",
            {"make": "Toyota", "model": "Camry", "price": 26000, "days_on_lot": 67},
        ),
    ]
    collection.add(
        documents=[v[0] for v in vehicles],
        metadatas=[v[1] for v in vehicles],
        ids=[f"vehicle_{i}" for i in range(len(vehicles))],
    )
    return collection


def _demo_inventory_stats(metric: str) -> str:
    """Return demo stats when DB is not available."""
    return (
        "Demo Inventory Overview\n" + "=" * 40 + "\n"
        "Total Vehicles:     247\n"
        "Avg Days on Lot:    24.3 days\n"
        "Avg Price:          $34,200\n"
        "Price Range:        $15,500 - $68,000\n"
        "At Risk (60+ days): 18 vehicles\n"
        "(Note: Live DB not connected — showing demo data)"
    )


def _format_df_output(title: str, df: pd.DataFrame) -> str:
    return f"{title}\n{'='*40}\n{df.to_string(index=False)}"


# ── Tool registry (used by graph.py) ─────────────────────────────────────────
ALL_TOOLS = [
    search_inventory,
    predict_days_on_lot,
    get_rep_archetypes,
    score_customer_sentiment,
    query_inventory_stats,
]
