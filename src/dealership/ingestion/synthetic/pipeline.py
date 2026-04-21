"""
Pipeline orchestrator — generate → validate → write Parquet.

Public files (data/raw/synthetic/):   no latent columns
Debug files  (data/debug/synthetic/): includes _* latent columns
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from dealership.common.config import get_settings
from dealership.ingestion.synthetic.generators import (
    generate_customers,
    generate_reps,
    generate_sales,
    generate_vehicles,
)

settings = get_settings()
_LATENT_PREFIX = "_"


def _strip_latent(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith(_LATENT_PREFIX)])


def _write_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy", engine="pyarrow")
    logger.info(f"  Wrote {label}: {len(df):,} rows -> {path.name} ({path.stat().st_size//1024} KB)")


def _integrity_checks(reps, customers, vehicles, sales) -> list[str]:
    errors: list[str] = []
    rep_ids = set(reps["rep_id"])
    cust_ids = set(customers["customer_id"])
    veh_ids = set(vehicles["vehicle_id"])
    if bad := set(sales["rep_id"]) - rep_ids:
        errors.append(f"Sales reference {len(bad)} unknown rep_ids")
    if bad := set(sales["customer_id"]) - cust_ids:
        errors.append(f"Sales reference {len(bad)} unknown customer_ids")
    if bad := set(sales["vehicle_id"]) - veh_ids:
        errors.append(f"Sales reference {len(bad)} unknown vehicle_ids")
    merged = sales.merge(vehicles[["vehicle_id", "cost_basis"]], on="vehicle_id", how="left")
    underwater = (merged["sale_price"] < merged["cost_basis"]).mean()
    if underwater > 0.10:
        errors.append(f"{underwater:.1%} of sales below cost basis (threshold 10%)")
    return errors


def run_pipeline(
    *,
    seed: int | None = None,
    num_reps: int | None = None,
    num_customers: int | None = None,
    num_vehicles: int | None = None,
    num_sales: int | None = None,
    raw_path: Path | None = None,
    debug_path: Path | None = None,
    skip_debug: bool = False,
) -> dict[str, Any]:
    """Execute the full synthetic data pipeline."""
    seed = seed if seed is not None else settings.synthetic_seed
    raw_out = raw_path or settings.raw_path()
    dbg_out = debug_path or settings.debug_path()

    logger.info("=" * 55)
    logger.info("Dealership Synthetic Data Pipeline — START")
    logger.info(f"  seed={seed}")
    logger.info("=" * 55)

    master = np.random.default_rng(seed)

    def child() -> np.random.Generator:
        return np.random.default_rng(int(master.integers(0, 2**31)))

    reps = generate_reps(n=num_reps, rng=child())
    customers = generate_customers(n=num_customers, rng=child())
    vehicles = generate_vehicles(n=num_vehicles, rng=child())
    sales = generate_sales(vehicles, customers, reps, n=num_sales, rng=child())

    logger.info("Running integrity checks ...")
    errors = _integrity_checks(reps, customers, vehicles, sales)
    if errors:
        for e in errors:
            logger.error(f"  x {e}")
        raise RuntimeError(f"Pipeline integrity checks failed ({len(errors)} errors)")
    logger.success("  All integrity checks passed")

    tables = {"reps": reps, "customers": customers, "vehicles": vehicles, "sales": sales}

    logger.info("Writing public Parquet ...")
    for name, df in tables.items():
        _write_parquet(_strip_latent(df), raw_out / f"{name}.parquet", name)

    if not skip_debug:
        logger.info("Writing debug Parquet (with latent columns) ...")
        for name, df in tables.items():
            _write_parquet(df, dbg_out / f"{name}_with_latent.parquet", name)

    sales_dt = pd.to_datetime(sales["sale_date"])
    metadata: dict[str, Any] = {
        "pipeline_version": "1.0.0",
        "generated_at_utc": datetime.now(tz=UTC).isoformat(),
        "seed": seed,
        "record_counts": {k: len(v) for k, v in tables.items()},
        "date_range": {"start": settings.synthetic_date_start, "end": settings.synthetic_date_end},
        "archetype_mix": reps["_archetype"].value_counts().to_dict(),
        "condition_mix": vehicles["condition"].value_counts().to_dict(),
        "avg_sale_price": round(float(sales["sale_price"].mean()), 2),
        "avg_gross_profit": round(float(sales["gross_profit"].mean()), 2),
        "avg_days_on_lot": round(float(sales["days_on_lot"].mean()), 1),
        "december_avg_discount_pct": round(float(sales[sales_dt.dt.month == 12]["discount_pct"].mean()), 4),
        "march_avg_discount_pct": round(float(sales[sales_dt.dt.month == 3]["discount_pct"].mean()), 4),
    }
    meta_path = raw_out / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"  Metadata -> {meta_path}")
    logger.success("Pipeline complete")
    return metadata
