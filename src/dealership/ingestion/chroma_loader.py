"""
ChromaDB ingestion loader for the Dealership Intelligence Agent.

Loads vehicle inventory and sales summaries into ChromaDB as
vector embeddings for semantic search.

Collections
-----------
vehicles    : One document per vehicle with make/model/price/condition
sales_facts : One document per sale with rep/customer/vehicle summary

Usage::

    from dealership.ingestion.chroma_loader import load_all_to_chroma
    load_all_to_chroma()
"""

from __future__ import annotations

import chromadb
import duckdb
from loguru import logger

from dealership.common.config import get_settings

settings = get_settings()

# Collection names
VEHICLES_COLLECTION = "vehicles"
SALES_COLLECTION = "sales_facts"


def get_chroma_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    chroma_path = "data/chroma"
    import os

    os.makedirs(chroma_path, exist_ok=True)
    client = chromadb.PersistentClient(path=chroma_path)
    logger.debug(f"ChromaDB client ready at {chroma_path}")
    return client


def _build_vehicle_doc(row: tuple) -> tuple[str, str, dict]:
    """
    Build a ChromaDB document from a vehicle row.

    Returns (id, document_text, metadata).
    """
    (
        vehicle_id,
        vin,
        make,
        model,
        year,
        trim,
        color,
        segment,
        condition,
        mileage,
        msrp,
        cost_basis,
        asking_price,
        arrived_date,
    ) = row

    doc = (
        f"{year} {make} {model} {trim}. "
        f"Condition: {condition}. "
        f"Color: {color}. "
        f"Segment: {segment}. "
        f"Mileage: {mileage:,} miles. "
        f"Asking price: ${asking_price:,}. "
        f"MSRP: ${msrp:,}. "
        f"Arrived: {arrived_date}."
    )

    metadata = {
        "vehicle_id": str(vehicle_id),
        "vin": str(vin),
        "make": str(make),
        "model": str(model),
        "year": int(year),
        "trim": str(trim),
        "color": str(color),
        "segment": str(segment),
        "condition": str(condition),
        "mileage": int(mileage),
        "asking_price": int(asking_price),
        "msrp": int(msrp),
    }

    return str(vehicle_id), doc, metadata


def _build_sale_doc(row: tuple) -> tuple[str, str, dict]:
    """
    Build a ChromaDB document from a sale row.

    Returns (id, document_text, metadata).
    """
    (
        sale_id,
        sale_date,
        make,
        model,
        condition,
        rep_name,
        customer_name,
        sale_price,
        gross_profit,
        discount_pct,
        days_on_lot,
        aging_bucket,
        margin_tier,
        satisfaction,
        financing_type,
        territory,
    ) = row

    doc = (
        f"Sale on {sale_date}: {make} {model} ({condition}). "
        f"Rep: {rep_name} ({territory} territory). "
        f"Customer: {customer_name}. "
        f"Sale price: ${sale_price:,}. "
        f"Gross profit: ${gross_profit:,}. "
        f"Discount: {discount_pct:.1%}. "
        f"Days on lot: {days_on_lot} ({aging_bucket}). "
        f"Margin: {margin_tier}. "
        f"Satisfaction: {satisfaction}/5. "
        f"Financing: {financing_type}."
    )

    metadata = {
        "sale_id": str(sale_id),
        "sale_date": str(sale_date),
        "make": str(make),
        "model": str(model),
        "condition": str(condition),
        "rep_name": str(rep_name),
        "territory": str(territory),
        "sale_price": int(sale_price),
        "gross_profit": int(gross_profit),
        "days_on_lot": int(days_on_lot),
        "aging_bucket": str(aging_bucket),
        "margin_tier": str(margin_tier),
        "satisfaction": float(satisfaction),
    }

    return str(sale_id), doc, metadata


def load_vehicles(
    client: chromadb.ClientAPI,
    conn: duckdb.DuckDBPyConnection,
    batch_size: int = 100,
) -> int:
    """
    Load vehicle inventory into ChromaDB.

    Parameters
    ----------
    client : chromadb.ClientAPI
        ChromaDB client.
    conn : duckdb.DuckDBPyConnection
        DuckDB connection.
    batch_size : int
        Number of documents per upsert batch.

    Returns
    -------
    int
        Number of documents loaded.
    """
    collection = client.get_or_create_collection(
        name=VEHICLES_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    rows = conn.execute("""
        SELECT
            vehicle_id, vin, make, model, year, trim,
            color, segment, condition, mileage,
            msrp, cost_basis, asking_price, arrived_date
        FROM raw.vehicles
    """).fetchall()

    ids, docs, metas = [], [], []
    for row in rows:
        vid, doc, meta = _build_vehicle_doc(row)
        ids.append(vid)
        docs.append(doc)
        metas.append(meta)

    # Batch upsert
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=docs[i : i + batch_size],
            metadatas=metas[i : i + batch_size],
        )

    logger.info(f"  Loaded {len(ids)} vehicles into ChromaDB")
    return len(ids)


def load_sales(
    client: chromadb.ClientAPI,
    conn: duckdb.DuckDBPyConnection,
    batch_size: int = 100,
) -> int:
    """
    Load sales facts into ChromaDB.

    Parameters
    ----------
    client : chromadb.ClientAPI
        ChromaDB client.
    conn : duckdb.DuckDBPyConnection
        DuckDB connection.
    batch_size : int
        Number of documents per upsert batch.

    Returns
    -------
    int
        Number of documents loaded.
    """
    collection = client.get_or_create_collection(
        name=SALES_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    rows = conn.execute("""
        SELECT
            sale_id, sale_date, make, model, vehicle_condition,
            rep_name, customer_name, sale_price,
            gross_profit, discount_pct, days_on_lot,
            aging_bucket, margin_tier,
            customer_satisfaction_score,
            financing_type, rep_territory
        FROM main_marts.fct_sales
    """).fetchall()

    ids, docs, metas = [], [], []
    for row in rows:
        sid, doc, meta = _build_sale_doc(row)
        ids.append(sid)
        docs.append(doc)
        metas.append(meta)

    # Batch upsert
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            documents=docs[i : i + batch_size],
            metadatas=metas[i : i + batch_size],
        )

    logger.info(f"  Loaded {len(ids)} sales into ChromaDB")
    return len(ids)


def load_all_to_chroma(
    db_path: str | None = None,
) -> dict[str, int]:
    """
    Load all data into ChromaDB.

    Parameters
    ----------
    db_path : str, optional
        Path to DuckDB file.

    Returns
    -------
    dict[str, int]
        Mapping of collection name to document count.
    """
    logger.info("=" * 50)
    logger.info("ChromaDB Loader — START")
    logger.info("=" * 50)

    db = db_path or str(settings.duckdb_file())
    conn = duckdb.connect(db, read_only=True)
    client = get_chroma_client()

    counts: dict[str, int] = {}
    counts[VEHICLES_COLLECTION] = load_vehicles(client, conn)
    counts[SALES_COLLECTION] = load_sales(client, conn)

    conn.close()

    logger.success("ChromaDB load complete — " + ", ".join(f"{k}:{v}" for k, v in counts.items()))
    logger.info("=" * 50)
    return counts


def semantic_search(
    query: str,
    collection_name: str = VEHICLES_COLLECTION,
    n_results: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """
    Run a semantic search against a ChromaDB collection.

    Parameters
    ----------
    query : str
        Natural language search query.
    collection_name : str
        Collection to search.
    n_results : int
        Number of results to return.
    where : dict, optional
        Metadata filter (ChromaDB where clause).

    Returns
    -------
    list[dict]
        List of results with document text, metadata, and distance.
    """
    client = get_chroma_client()
    collection = client.get_collection(name=collection_name)

    kwargs: dict = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    results = collection.query(**kwargs)

    output = []
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            strict=False,
        )
    ):
        output.append(
            {
                "rank": i + 1,
                "document": doc,
                "metadata": meta,
                "distance": round(dist, 4),
            }
        )

    return output
