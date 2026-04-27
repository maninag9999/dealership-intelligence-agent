#!/usr/bin/env python
"""
API launcher for the Dealership Intelligence Agent.

Loads ChromaDB data then starts the FastAPI server via uvicorn.

Usage::

    uv run python scripts/start_api.py
    uv run python scripts/start_api.py --port 8080
    uv run python scripts/start_api.py --skip-chroma
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
import uvicorn
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dealership.ingestion.chroma_loader import load_all_to_chroma

app = typer.Typer(name="start-api", add_completion=False)


def _configure_logging(level: str) -> None:
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)


@app.command()
def main(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind"),
    port: int = typer.Option(8000, "--port", help="Port to bind"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
    skip_chroma: bool = typer.Option(False, "--skip-chroma", help="Skip ChromaDB loading"),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Start the Dealership Intelligence Agent API."""
    _configure_logging(log_level.upper())

    # Load ChromaDB first
    if not skip_chroma:
        logger.info("Loading ChromaDB ...")
        try:
            counts = load_all_to_chroma()
            logger.success(f"ChromaDB ready — {counts}")
        except Exception as exc:
            logger.warning(f"ChromaDB load failed (continuing anyway): {exc}")
    else:
        logger.info("Skipping ChromaDB load (--skip-chroma)")

    # Start API
    logger.info(f"Starting API on http://{host}:{port}")
    logger.info(f"Docs available at http://localhost:{port}/docs")

    uvicorn.run(
        "dealership.api.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level.lower(),
        app_dir=str(Path(__file__).resolve().parent.parent / "src"),
    )


if __name__ == "__main__":
    app()
