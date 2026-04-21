#!/usr/bin/env python
"""Sanity checks for generated synthetic data. Exits 1 on failure."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from dealership.common.config import get_settings

app = typer.Typer(add_completion=False)
_P, _F = "PASS", "FAIL"


def _check(name: str, ok: bool, detail: str) -> bool:
    icon = "OK" if ok else "XX"
    (logger.success if ok else logger.error)(f"[{icon}] {name} -> {detail}")
    return ok


@app.command()
def main(
    raw_path: Path = typer.Option(None, "--raw-path"),  # noqa: B008,
    price_aging_threshold: float = typer.Option(0.10, "--r-threshold"),
    dec_delta: float = typer.Option(0.010, "--dec-delta"),
    rep_iqr: float = typer.Option(500.0, "--rep-iqr"),
) -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
        level="DEBUG",
    )

    s = get_settings()
    raw_dir = raw_path or s.raw_path()
    logger.info(f"Checking data at: {raw_dir}")

    try:
        reps = pd.read_parquet(raw_dir / "reps.parquet")
        customers = pd.read_parquet(raw_dir / "customers.parquet")
        vehicles = pd.read_parquet(raw_dir / "vehicles.parquet")
        sales = pd.read_parquet(raw_dir / "sales.parquet")
    except FileNotFoundError as e:
        logger.error(f"Missing file: {e}. Run generate_synthetic_data.py first.")
        raise typer.Exit(1) from None

    logger.info(f"reps:{len(reps):,}  customers:{len(customers):,}  vehicles:{len(vehicles):,}  sales:{len(sales):,}")

    results = []

    # 1. No latent cols in public files
    for name, df in [
        ("reps", reps),
        ("customers", customers),
        ("vehicles", vehicles),
        ("sales", sales),
    ]:
        latent = [c for c in df.columns if c.startswith("_")]
        results.append(
            _check(
                f"No latent cols in {name}.parquet",
                len(latent) == 0,
                "clean" if not latent else f"found: {latent}",
            )
        )

    # 2. Referential integrity
    bad_r = set(sales["rep_id"]) - set(reps["rep_id"])
    bad_c = set(sales["customer_id"]) - set(customers["customer_id"])
    bad_v = set(sales["vehicle_id"]) - set(vehicles["vehicle_id"])
    all_ok = not (bad_r or bad_c or bad_v)
    results.append(
        _check(
            "Referential integrity",
            all_ok,
            "all FKs resolve" if all_ok else f"orphans rep={len(bad_r)} cust={len(bad_c)} veh={len(bad_v)}",
        )
    )

    # 3. Price <-> aging correlation
    r = sales["asking_price_at_sale"].corr(sales["days_on_lot"])
    results.append(
        _check(
            "Price-aging correlation",
            r > price_aging_threshold,
            f"Pearson r={r:.4f} (threshold>{price_aging_threshold})",
        )
    )

    # 4. December discount > March
    dt = pd.to_datetime(sales["sale_date"])
    dec_d = sales[dt.dt.month == 12]["discount_pct"].mean()
    mar_d = sales[dt.dt.month == 3]["discount_pct"].mean()
    delta = dec_d - mar_d
    results.append(
        _check(
            "December > March discount",
            delta > dec_delta,
            f"Dec={dec_d:.4f} Mar={mar_d:.4f} delta={delta:.4f} (threshold>{dec_delta})",
        )
    )

    # 5. Rep margin IQR
    per_rep = sales.groupby("rep_id")["gross_profit"].median()
    q25, q75 = per_rep.quantile(0.25), per_rep.quantile(0.75)
    iqr = q75 - q25
    results.append(
        _check(
            "Rep margin spread IQR",
            iqr > rep_iqr,
            f"IQR=${iqr:,.0f} Q25=${q25:,.0f} Q75=${q75:,.0f} (threshold>${rep_iqr:,.0f})",
        )
    )

    logger.info("-" * 50)
    passed = sum(results)
    if passed == len(results):
        logger.success(f"All {passed}/{len(results)} checks passed!")
        raise typer.Exit(0)
    else:
        logger.error(f"{len(results)-passed}/{len(results)} checks FAILED")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
