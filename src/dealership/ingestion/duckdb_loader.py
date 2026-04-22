"""
DuckDB ingestion loader for the Dealership Intelligence Agent.

Reads public Parquet files from data/raw/synthetic/ and loads them
into DuckDB under the ``raw`` schema.  Safe to re-run — tables are
replaced each time (full-refresh pattern for synthetic data).

Usage::

    from dealership.ingestion.duckdb_loader import load_all
    load_all()
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger

from dealership.common.config import get_settings

settings = get_settings()

# Tables to load — order matters for FK integrity checks later
_TABLES: list[tuple[str, str]] = [
    ("reps", "reps.parquet"),
    ("customers", "customers.parquet"),
    ("vehicles", "vehicles.parquet"),
    ("sales", "sales.parquet"),
]


def get_connection(db_path: Path | None = None) -> duckdb.DuckDBPyConnection:
    """
    Return a DuckDB connection to the warehouse file.

    Parameters
    ----------
    db_path : Path, optional
        Path to the DuckDB file.  Defaults to ``settings.duckdb_file()``.
    """
    db_path = db_path or settings.duckdb_file()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Connecting to DuckDB at {db_path}")
    return duckdb.connect(str(db_path))


def load_table(
    conn: duckdb.DuckDBPyConnection,
    table_name: str,
    parquet_path: Path,
) -> int:
    """
    Load a single Parquet file into DuckDB as ``raw.<table_name>``.

    Uses CREATE OR REPLACE so re-runs are safe.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    table_name : str
        Target table name inside the ``raw`` schema.
    parquet_path : Path
        Absolute path to the source Parquet file.

    Returns
    -------
    int
        Row count of the loaded table.
    """
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Parquet file not found: {parquet_path}\n" "Run 'uv run python scripts/generate_synthetic_data.py' first."
        )

    sql = f"""
        CREATE OR REPLACE TABLE raw.{table_name} AS
        SELECT * FROM read_parquet('{parquet_path.as_posix()}')
    """
    conn.execute(sql)
    row_count = conn.execute(f"SELECT COUNT(*) FROM raw.{table_name}").fetchone()[0]
    logger.info(f"  Loaded raw.{table_name}: {row_count:,} rows")
    return row_count


def load_all(
    db_path: Path | None = None,
    raw_path: Path | None = None,
) -> dict[str, int]:
    """
    Load all synthetic Parquet files into DuckDB.

    Creates the ``raw`` schema if it does not exist, then loads
    reps, customers, vehicles, and sales tables.

    Parameters
    ----------
    db_path : Path, optional
        DuckDB warehouse file path.
    raw_path : Path, optional
        Directory containing the public Parquet files.

    Returns
    -------
    dict[str, int]
        Mapping of table name to row count.
    """
    raw_dir = raw_path or settings.raw_path()
    conn = get_connection(db_path)

    logger.info("=" * 50)
    logger.info("DuckDB Loader — START")
    logger.info(f"  source : {raw_dir}")
    logger.info(f"  target : {settings.duckdb_file()}")
    logger.info("=" * 50)

    # Create raw schema
    conn.execute("CREATE SCHEMA IF NOT EXISTS raw")
    logger.info("Schema 'raw' ready")

    counts: dict[str, int] = {}
    for table_name, filename in _TABLES:
        parquet_path = raw_dir / filename
        counts[table_name] = load_table(conn, table_name, parquet_path)

    # Quick referential integrity check inside DuckDB
    logger.info("Running in-database integrity check ...")
    orphan_reps = conn.execute("""
        SELECT COUNT(*) FROM raw.sales s
        LEFT JOIN raw.reps r ON s.rep_id = r.rep_id
        WHERE r.rep_id IS NULL
    """).fetchone()[0]

    orphan_custs = conn.execute("""
        SELECT COUNT(*) FROM raw.sales s
        LEFT JOIN raw.customers c ON s.customer_id = c.customer_id
        WHERE c.customer_id IS NULL
    """).fetchone()[0]

    orphan_vehs = conn.execute("""
        SELECT COUNT(*) FROM raw.sales s
        LEFT JOIN raw.vehicles v ON s.vehicle_id = v.vehicle_id
        WHERE v.vehicle_id IS NULL
    """).fetchone()[0]

    if any([orphan_reps, orphan_custs, orphan_vehs]):
        raise RuntimeError(
            f"Integrity check failed — orphan FKs: "
            f"reps={orphan_reps} customers={orphan_custs} vehicles={orphan_vehs}"
        )
    logger.success("  Integrity check passed — all FKs resolve")

    # Summary stats
    avg_sale = conn.execute("SELECT ROUND(AVG(sale_price), 2) FROM raw.sales").fetchone()[0]
    avg_gross = conn.execute("SELECT ROUND(AVG(gross_profit), 2) FROM raw.sales").fetchone()[0]
    logger.success(f"Load complete — avg sale ${avg_sale:,.0f}, avg gross ${avg_gross:,.0f}")

    conn.close()

    logger.info("=" * 50)
    logger.success("DuckDB Loader — DONE")
    logger.info("=" * 50)

    return counts
