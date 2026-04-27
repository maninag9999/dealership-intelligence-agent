"""
Sales routes for the Dealership Intelligence Agent API.

Exposes endpoints for querying sales data from DuckDB fct_sales mart.
"""

from __future__ import annotations

from typing import Any

import duckdb
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from dealership.common.config import get_settings

settings = get_settings()
router = APIRouter()


def _get_conn() -> duckdb.DuckDBPyConnection:
    """Return a read-only DuckDB connection."""
    return duckdb.connect(str(settings.duckdb_file()), read_only=True)


# ------------------------------------------------------------------
# Summary endpoint
# ------------------------------------------------------------------


@router.get("/summary")
def sales_summary() -> dict[str, Any]:
    """
    Return high-level sales summary metrics.

    Includes total sales, revenue, avg gross profit,
    avg days on lot, and avg satisfaction score.
    """
    logger.info("GET /api/v1/sales/summary")
    try:
        conn = _get_conn()
        row = conn.execute("""
            SELECT
                COUNT(*)                          AS total_sales,
                ROUND(SUM(sale_price), 2)         AS total_revenue,
                ROUND(AVG(sale_price), 2)         AS avg_sale_price,
                ROUND(AVG(gross_profit), 2)       AS avg_gross_profit,
                ROUND(AVG(discount_pct), 4)       AS avg_discount_pct,
                ROUND(AVG(days_on_lot), 1)        AS avg_days_on_lot,
                ROUND(AVG(customer_satisfaction_score), 2) AS avg_satisfaction
            FROM main_marts.fct_sales
        """).fetchone()
        conn.close()

        return {
            "total_sales": row[0],
            "total_revenue": row[1],
            "avg_sale_price": row[2],
            "avg_gross_profit": row[3],
            "avg_discount_pct": row[4],
            "avg_days_on_lot": row[5],
            "avg_satisfaction": row[6],
        }
    except Exception as exc:
        logger.error(f"sales_summary error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Monthly trends
# ------------------------------------------------------------------


@router.get("/monthly")
def sales_by_month() -> list[dict[str, Any]]:
    """
    Return sales metrics grouped by year and month.

    Useful for trend analysis and December discount spike visualisation.
    """
    logger.info("GET /api/v1/sales/monthly")
    try:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT
                sale_year,
                sale_month,
                COUNT(*)                          AS total_sales,
                ROUND(AVG(sale_price), 2)         AS avg_sale_price,
                ROUND(AVG(gross_profit), 2)       AS avg_gross_profit,
                ROUND(AVG(discount_pct), 4)       AS avg_discount_pct,
                ROUND(AVG(days_on_lot), 1)        AS avg_days_on_lot
            FROM main_marts.fct_sales
            GROUP BY sale_year, sale_month
            ORDER BY sale_year, sale_month
        """).fetchall()
        conn.close()

        return [
            {
                "year": r[0],
                "month": r[1],
                "total_sales": r[2],
                "avg_sale_price": r[3],
                "avg_gross_profit": r[4],
                "avg_discount_pct": r[5],
                "avg_days_on_lot": r[6],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"sales_by_month error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Top vehicles
# ------------------------------------------------------------------


@router.get("/top-vehicles")
def top_vehicles(
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict[str, Any]]:
    """
    Return top selling vehicle make/model combinations.

    Parameters
    ----------
    limit : int
        Number of results to return (1-50, default 10).
    """
    logger.info(f"GET /api/v1/sales/top-vehicles?limit={limit}")
    try:
        conn = _get_conn()
        rows = conn.execute(f"""
            SELECT
                make,
                model,
                vehicle_segment,
                COUNT(*)                          AS total_sales,
                ROUND(AVG(sale_price), 2)         AS avg_sale_price,
                ROUND(AVG(gross_profit), 2)       AS avg_gross_profit,
                ROUND(AVG(days_on_lot), 1)        AS avg_days_on_lot
            FROM main_marts.fct_sales
            GROUP BY make, model, vehicle_segment
            ORDER BY total_sales DESC
            LIMIT {limit}
        """).fetchall()
        conn.close()

        return [
            {
                "make": r[0],
                "model": r[1],
                "segment": r[2],
                "total_sales": r[3],
                "avg_sale_price": r[4],
                "avg_gross_profit": r[5],
                "avg_days_on_lot": r[6],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"top_vehicles error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Aging analysis
# ------------------------------------------------------------------


@router.get("/aging")
def aging_analysis() -> list[dict[str, Any]]:
    """
    Return days-on-lot distribution by aging bucket.

    Buckets: Fast (≤15d), Normal (≤30d), Slow (≤60d), Stale (>60d).
    """
    logger.info("GET /api/v1/sales/aging")
    try:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT
                aging_bucket,
                COUNT(*)                          AS total_sales,
                ROUND(AVG(days_on_lot), 1)        AS avg_days_on_lot,
                ROUND(AVG(gross_profit), 2)       AS avg_gross_profit,
                ROUND(AVG(discount_pct), 4)       AS avg_discount_pct
            FROM main_marts.fct_sales
            GROUP BY aging_bucket
            ORDER BY avg_days_on_lot
        """).fetchall()
        conn.close()

        return [
            {
                "aging_bucket": r[0],
                "total_sales": r[1],
                "avg_days_on_lot": r[2],
                "avg_gross_profit": r[3],
                "avg_discount_pct": r[4],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"aging_analysis error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Recent sales
# ------------------------------------------------------------------


@router.get("/recent")
def recent_sales(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    """
    Return most recent sales transactions.

    Parameters
    ----------
    limit : int
        Number of results (1-100, default 20).
    """
    logger.info(f"GET /api/v1/sales/recent?limit={limit}")
    try:
        conn = _get_conn()
        rows = conn.execute(f"""
            SELECT
                sale_id,
                sale_date,
                make,
                model,
                vehicle_condition,
                rep_name,
                customer_name,
                sale_price,
                gross_profit,
                discount_pct,
                days_on_lot,
                aging_bucket,
                margin_tier,
                customer_satisfaction_score
            FROM main_marts.fct_sales
            ORDER BY sale_date DESC
            LIMIT {limit}
        """).fetchall()
        conn.close()

        return [
            {
                "sale_id": r[0],
                "sale_date": str(r[1]),
                "make": r[2],
                "model": r[3],
                "condition": r[4],
                "rep_name": r[5],
                "customer_name": r[6],
                "sale_price": r[7],
                "gross_profit": r[8],
                "discount_pct": r[9],
                "days_on_lot": r[10],
                "aging_bucket": r[11],
                "margin_tier": r[12],
                "satisfaction": r[13],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"recent_sales error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
