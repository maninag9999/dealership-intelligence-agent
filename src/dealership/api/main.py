"""
FastAPI application entry point for the Dealership Intelligence Agent.

Registers all route modules and exposes a /health endpoint.

Usage::
    uv run python scripts/start_api.py
    # or directly:
    uv run uvicorn dealership.api.main:app --reload
"""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from dealership.api.routes import llm, reps, sales, search
from dealership.common.config import get_settings

settings = get_settings()

# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

app = FastAPI(
    title="Dealership Intelligence Agent API",
    description=(
        "Production-grade REST API for the Multi-Modal Dealership "
        "Intelligence Agent. Exposes sales analytics, rep performance, "
        "semantic search, and local LLM endpoints."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------
# Route registration
# ------------------------------------------------------------------

app.include_router(sales.router, prefix="/api/v1/sales", tags=["Sales"])
app.include_router(reps.router, prefix="/api/v1/reps", tags=["Reps"])
app.include_router(search.router, prefix="/api/v1/search", tags=["Search"])
app.include_router(llm.router, prefix="/api/v1/llm", tags=["LLM"])


# ------------------------------------------------------------------
# Health endpoint
# ------------------------------------------------------------------


@app.get("/health", tags=["Health"])
def health() -> dict:
    """
    Health check endpoint.

    Verifies:
    - API is running
    - DuckDB warehouse is reachable
    - Returns timestamp and record counts
    """
    status: dict = {
        "status": "ok",
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "version": "1.0.0",
        "warehouse": "unreachable",
        "counts": {},
    }

    try:
        conn = duckdb.connect(str(settings.duckdb_file()), read_only=True)
        counts = {
            "reps": conn.execute("SELECT COUNT(*) FROM raw.reps").fetchone()[0],
            "customers": conn.execute("SELECT COUNT(*) FROM raw.customers").fetchone()[0],
            "vehicles": conn.execute("SELECT COUNT(*) FROM raw.vehicles").fetchone()[0],
            "sales": conn.execute("SELECT COUNT(*) FROM raw.sales").fetchone()[0],
            "fct_sales": conn.execute("SELECT COUNT(*) FROM main_marts.fct_sales").fetchone()[0],
        }
        conn.close()
        status["warehouse"] = "ok"
        status["counts"] = counts
        logger.debug(f"Health check passed — counts: {counts}")
    except Exception as exc:
        logger.error(f"Health check — warehouse unreachable: {exc}")
        status["warehouse"] = f"error: {exc}"

    return status


@app.get("/", tags=["Health"])
def root() -> dict:
    """Root endpoint — redirects to docs."""
    return {
        "message": "Dealership Intelligence Agent API",
        "docs": "/docs",
        "health": "/health",
    }
