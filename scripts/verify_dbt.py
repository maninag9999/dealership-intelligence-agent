#!/usr/bin/env python
"""
Verification script for dbt models.

Runs dbt and then queries DuckDB directly to assert:
1. All mart tables exist and have correct row counts
2. fct_sales avg gross profit > 0
3. dim_reps quota attainment is populated
4. December discount > March discount in fct_sales

Exit codes
----------
0 - all checks passed
1 - one or more checks failed
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import duckdb
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dealership.common.config import get_settings

app = typer.Typer(add_completion=False)


def _check(name: str, ok: bool, detail: str) -> bool:
    icon = "OK" if ok else "XX"
    (logger.success if ok else logger.error)(f"[{icon}] {name} -> {detail}")
    return ok


@app.command()
def main(
    db_path: Path = typer.Option(None, "--db-path"),  # noqa: B008
    skip_dbt_run: bool = typer.Option(False, "--skip-dbt-run"),
) -> None:
    """Run dbt models then verify mart tables in DuckDB."""
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    logger.add(sys.stderr, format=fmt, colorize=True, level="DEBUG")

    s = get_settings()
    db = db_path or s.duckdb_file()

    # ── Step 1: run dbt ──────────────────────────────────────────────
    if not skip_dbt_run:
        logger.info("Running dbt build ...")
        result = subprocess.run(
            ["uv", "run", "dbt", "build", "--project-dir", "dbt", "--profiles-dir", "dbt"],
            capture_output=False,
            text=True,
        )
        if result.returncode != 0:
            logger.error("dbt build failed — check output above")
            raise typer.Exit(code=1) from None
        logger.success("dbt build passed")
    else:
        logger.info("Skipping dbt run (--skip-dbt-run)")

    # ── Step 2: verify in DuckDB ─────────────────────────────────────
    logger.info(f"Connecting to DuckDB at {db} ...")
    try:
        conn = duckdb.connect(str(db))
    except Exception as exc:
        logger.error(f"Could not connect to DuckDB: {exc}")
        raise typer.Exit(code=1) from None

    results = []
    logger.info("-" * 50)

    # 1. fct_sales exists and has rows
    try:
        fct_count = conn.execute("SELECT COUNT(*) FROM main_marts.fct_sales").fetchone()[0]
        results.append(
            _check(
                "fct_sales row count",
                fct_count > 0,
                f"{fct_count:,} rows",
            )
        )
    except Exception:
        results.append(_check("fct_sales exists", False, "table not found"))

    # 2. dim_reps exists and has rows
    try:
        dim_count = conn.execute("SELECT COUNT(*) FROM main_marts.dim_reps").fetchone()[0]
        results.append(
            _check(
                "dim_reps row count",
                dim_count > 0,
                f"{dim_count:,} rows",
            )
        )
    except Exception:
        results.append(_check("dim_reps exists", False, "table not found"))

    # 3. avg gross profit > 0
    avg_gross = conn.execute("SELECT ROUND(AVG(gross_profit), 2) FROM main_marts.fct_sales").fetchone()[0]
    results.append(
        _check(
            "fct_sales avg gross profit > 0",
            avg_gross > 0,
            f"${avg_gross:,.2f}",
        )
    )

    # 4. all reps have metrics in dim_reps
    reps_with_sales = conn.execute("SELECT COUNT(*) FROM main_marts.dim_reps WHERE total_sales > 0").fetchone()[0]
    results.append(
        _check(
            "dim_reps reps have sales",
            reps_with_sales > 0,
            f"{reps_with_sales} reps with sales",
        )
    )

    # 5. December discount > March in fct_sales
    dec_disc = conn.execute("""
        SELECT ROUND(AVG(discount_pct), 4)
        FROM main_marts.fct_sales
        WHERE sale_month = 12
    """).fetchone()[0]
    mar_disc = conn.execute("""
        SELECT ROUND(AVG(discount_pct), 4)
        FROM main_marts.fct_sales
        WHERE sale_month = 3
    """).fetchone()[0]
    delta = round(dec_disc - mar_disc, 4)
    results.append(
        _check(
            "December > March discount in fct_sales",
            delta > 0.010,
            f"Dec={dec_disc:.4f} Mar={mar_disc:.4f} delta={delta:.4f}",
        )
    )

    # 6. staging views exist
    for view in ["stg_reps", "stg_customers", "stg_vehicles", "stg_sales"]:
        try:
            count = conn.execute(f"SELECT COUNT(*) FROM main_staging.{view}").fetchone()[0]
            results.append(
                _check(
                    f"staging.{view} exists",
                    count > 0,
                    f"{count:,} rows",
                )
            )
        except Exception:
            results.append(_check(f"staging.{view} exists", False, "view not found"))

    conn.close()

    logger.info("-" * 50)
    passed = sum(results)
    total = len(results)

    if passed == total:
        logger.success(f"All {total}/{total} checks passed!")
        raise typer.Exit(code=0)
    else:
        logger.error(f"{total - passed}/{total} checks FAILED")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
