#!/usr/bin/env python
"""CLI entrypoint — generate synthetic dealership data.

Usage:
    uv run python scripts/generate_synthetic_data.py
    uv run python scripts/generate_synthetic_data.py --num-sales 2000 --seed 99
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dealership.common.config import get_settings
from dealership.ingestion.synthetic.pipeline import run_pipeline

app = typer.Typer(name="generate-synthetic-data", add_completion=False)


def _configure_logging(level: str, log_file: str) -> None:
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    logger.add(sys.stderr, format=fmt, level=level, colorize=True)
    logger.add(log_file, format=fmt, level=level, rotation="10 MB", retention="7 days")


@app.command()
def main(
    seed: int = typer.Option(None, "--seed"),
    num_reps: int = typer.Option(None, "--num-reps"),
    num_customers: int = typer.Option(None, "--num-customers"),
    num_vehicles: int = typer.Option(None, "--num-vehicles"),
    num_sales: int = typer.Option(None, "--num-sales"),
    skip_debug: bool = typer.Option(False, "--skip-debug"),
    log_level: str = typer.Option(None, "--log-level"),
) -> None:
    """Run the synthetic data generation pipeline."""
    s = get_settings()
    _configure_logging((log_level or s.log_level).upper(), s.log_file)

    try:
        meta = run_pipeline(
            seed=seed,
            num_reps=num_reps,
            num_customers=num_customers,
            num_vehicles=num_vehicles,
            num_sales=num_sales,
            skip_debug=skip_debug,
        )
    except RuntimeError as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise typer.Exit(code=1) from None

    typer.echo("")
    typer.secho("  Synthetic data generation complete", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Records   : {meta['record_counts']}")
    typer.echo(f"  Avg sale  : ${meta['avg_sale_price']:,.0f}")
    typer.echo(f"  Avg gross : ${meta['avg_gross_profit']:,.0f}")
    typer.echo(f"  Dec disc  : {meta['december_avg_discount_pct']:.2%}")
    typer.echo(f"  Mar disc  : {meta['march_avg_discount_pct']:.2%}")
    typer.echo(f"  Files     : {s.raw_path()}")


if __name__ == "__main__":
    app()
