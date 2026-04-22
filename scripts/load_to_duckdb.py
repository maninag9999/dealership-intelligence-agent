#!/usr/bin/env python
"""
CLI entrypoint — load synthetic Parquet files into DuckDB.

Usage::

    uv run python scripts/load_to_duckdb.py
    uv run python scripts/load_to_duckdb.py --db-path data/warehouse/dealership.duckdb
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dealership.common.config import get_settings
from dealership.ingestion.duckdb_loader import load_all

app = typer.Typer(name="load-to-duckdb", add_completion=False)


def _configure_logging(level: str, log_file: str) -> None:
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)
    logger.add(log_file, format=fmt, level=level, rotation="10 MB", retention="7 days")


@app.command()
def main(
    db_path: Path = typer.Option(  # noqa: B008
        None,
        "--db-path",
        help="Path to DuckDB file (default: from .env)",
        dir_okay=False,
    ),
    raw_path: Path = typer.Option(  # noqa: B008
        None,
        "--raw-path",
        help="Directory containing public Parquet files",
        dir_okay=True,
        file_okay=False,
    ),
    log_level: str = typer.Option(
        None,
        "--log-level",
        help="Logging level (DEBUG/INFO/WARNING/ERROR)",
    ),
) -> None:
    """Load synthetic Parquet files into DuckDB warehouse."""
    s = get_settings()
    _configure_logging((log_level or s.log_level).upper(), s.log_file)

    try:
        counts = load_all(db_path=db_path, raw_path=raw_path)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error(f"Load failed: {exc}")
        raise typer.Exit(code=1) from None

    typer.echo("")
    typer.secho("  DuckDB load complete", fg=typer.colors.GREEN, bold=True)
    for table, count in counts.items():
        typer.echo(f"  raw.{table:<12}: {count:,} rows")
    typer.echo(f"  Database    : {db_path or s.duckdb_file()}")


if __name__ == "__main__":
    app()
