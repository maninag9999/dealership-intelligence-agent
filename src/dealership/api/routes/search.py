"""
Semantic search routes for the Dealership Intelligence Agent API.

Uses ChromaDB to perform natural language search over
vehicle inventory and sales facts.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel

from dealership.ingestion.chroma_loader import (
    SALES_COLLECTION,
    VEHICLES_COLLECTION,
    semantic_search,
)

router = APIRouter()


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Request body for semantic search."""

    query: str
    n_results: int = 5
    collection: str = VEHICLES_COLLECTION


class SearchResult(BaseModel):
    """Single search result."""

    rank: int
    document: str
    metadata: dict[str, Any]
    distance: float


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.get("/vehicles")
def search_vehicles(
    q: str = Query(..., description="Natural language search query"),
    n: int = Query(default=5, ge=1, le=20),
    condition: str = Query(default=None, description="Filter: New | Used | Certified Pre-Owned"),
    segment: str = Query(default=None, description="Filter: SUV | Sedan | Truck | Coupe"),
) -> list[dict[str, Any]]:
    """
    Semantic search over vehicle inventory.

    Examples
    --------
    - "reliable Toyota SUV under 30k miles"
    - "affordable sedan good for first time buyer"
    - "truck with low mileage"

    Parameters
    ----------
    q : str
        Natural language query.
    n : int
        Number of results (1-20).
    condition : str, optional
        Filter by vehicle condition.
    segment : str, optional
        Filter by vehicle segment.
    """
    logger.info(f"GET /api/v1/search/vehicles?q={q}&n={n}")
    try:
        where: dict | None = None
        if condition and segment:
            where = {"$and": [{"condition": condition}, {"segment": segment}]}
        elif condition:
            where = {"condition": condition}
        elif segment:
            where = {"segment": segment}

        results = semantic_search(
            query=q,
            collection_name=VEHICLES_COLLECTION,
            n_results=n,
            where=where,
        )
        return results
    except Exception as exc:
        logger.error(f"search_vehicles error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/sales")
def search_sales(
    q: str = Query(..., description="Natural language search query"),
    n: int = Query(default=5, ge=1, le=20),
    margin_tier: str = Query(default=None, description="Filter: High | Medium | Low | Loss"),
    aging_bucket: str = Query(default=None, description="Filter: Fast | Normal | Slow | Stale"),
) -> list[dict[str, Any]]:
    """
    Semantic search over sales transactions.

    Examples
    --------
    - "high margin truck sales in December"
    - "struggling rep with stale inventory"
    - "satisfied customer who traded in"

    Parameters
    ----------
    q : str
        Natural language query.
    n : int
        Number of results (1-20).
    margin_tier : str, optional
        Filter by margin tier.
    aging_bucket : str, optional
        Filter by aging bucket.
    """
    logger.info(f"GET /api/v1/search/sales?q={q}&n={n}")
    try:
        where: dict | None = None
        if margin_tier and aging_bucket:
            where = {
                "$and": [
                    {"margin_tier": margin_tier},
                    {"aging_bucket": aging_bucket},
                ]
            }
        elif margin_tier:
            where = {"margin_tier": margin_tier}
        elif aging_bucket:
            where = {"aging_bucket": aging_bucket}

        results = semantic_search(
            query=q,
            collection_name=SALES_COLLECTION,
            n_results=n,
            where=where,
        )
        return results
    except Exception as exc:
        logger.error(f"search_sales error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/")
def search_any(request: SearchRequest) -> list[dict[str, Any]]:
    """
    Generic semantic search endpoint.

    Accepts a JSON body with query, n_results, and collection.
    Useful for the LangGraph agent to call programmatically.
    """
    logger.info(f"POST /api/v1/search/ query={request.query}")
    try:
        results = semantic_search(
            query=request.query,
            collection_name=request.collection,
            n_results=request.n_results,
        )
        return results
    except Exception as exc:
        logger.error(f"search_any error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
