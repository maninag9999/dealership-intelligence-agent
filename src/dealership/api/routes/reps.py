"""
Rep performance routes for the Dealership Intelligence Agent API.

Exposes endpoints for querying rep performance data from dim_reps mart.
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
# Leaderboard
# ------------------------------------------------------------------


@router.get("/leaderboard")
def rep_leaderboard(
    metric: str = Query(
        default="total_gross_profit",
        description="Sort metric: total_gross_profit | total_sales | avg_satisfaction_score",
    ),
    limit: int = Query(default=10, ge=1, le=20),
) -> list[dict[str, Any]]:
    """
    Return rep leaderboard sorted by chosen metric.

    Parameters
    ----------
    metric : str
        Column to sort by.
    limit : int
        Number of reps to return.
    """
    allowed = {
        "total_gross_profit",
        "total_sales",
        "avg_satisfaction_score",
        "avg_gross_profit",
        "quota_attainment_2yr",
    }
    if metric not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"metric must be one of {allowed}",
        )

    logger.info(f"GET /api/v1/reps/leaderboard?metric={metric}&limit={limit}")
    try:
        conn = _get_conn()
        rows = conn.execute(f"""
            SELECT
                rep_id,
                rep_name,
                territory,
                total_sales,
                total_revenue,
                total_gross_profit,
                avg_gross_profit,
                avg_discount_pct,
                avg_days_on_lot,
                avg_satisfaction_score,
                quota_attainment_2yr,
                high_margin_sales,
                loss_sales
            FROM main_marts.dim_reps
            ORDER BY {metric} DESC
            LIMIT {limit}
        """).fetchall()
        conn.close()

        return [
            {
                "rep_id": r[0],
                "rep_name": r[1],
                "territory": r[2],
                "total_sales": r[3],
                "total_revenue": r[4],
                "total_gross_profit": r[5],
                "avg_gross_profit": r[6],
                "avg_discount_pct": r[7],
                "avg_days_on_lot": r[8],
                "avg_satisfaction": r[9],
                "quota_attainment_2yr": r[10],
                "high_margin_sales": r[11],
                "loss_sales": r[12],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"rep_leaderboard error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Individual rep
# ------------------------------------------------------------------


@router.get("/{rep_id}")
def get_rep(rep_id: str) -> dict[str, Any]:
    """
    Return full performance profile for a single rep.

    Parameters
    ----------
    rep_id : str
        UUID of the sales rep.
    """
    logger.info(f"GET /api/v1/reps/{rep_id}")
    try:
        conn = _get_conn()
        row = conn.execute(
            """
            SELECT
                rep_id,
                rep_name,
                territory,
                hire_date,
                years_experience,
                monthly_quota_usd,
                total_sales,
                total_revenue,
                total_gross_profit,
                avg_gross_profit,
                avg_discount_pct,
                avg_days_on_lot,
                avg_satisfaction_score,
                dec_avg_discount_pct,
                dec_avg_gross_profit,
                high_margin_sales,
                loss_sales,
                stale_sales,
                quota_attainment_2yr
            FROM main_marts.dim_reps
            WHERE rep_id = ?
        """,
            [rep_id],
        ).fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"Rep {rep_id} not found")

        return {
            "rep_id": row[0],
            "rep_name": row[1],
            "territory": row[2],
            "hire_date": str(row[3]),
            "years_experience": row[4],
            "monthly_quota_usd": row[5],
            "total_sales": row[6],
            "total_revenue": row[7],
            "total_gross_profit": row[8],
            "avg_gross_profit": row[9],
            "avg_discount_pct": row[10],
            "avg_days_on_lot": row[11],
            "avg_satisfaction": row[12],
            "dec_avg_discount_pct": row[13],
            "dec_avg_gross_profit": row[14],
            "high_margin_sales": row[15],
            "loss_sales": row[16],
            "stale_sales": row[17],
            "quota_attainment_2yr": row[18],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"get_rep error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# Territory summary
# ------------------------------------------------------------------


@router.get("/territory/summary")
def territory_summary() -> list[dict[str, Any]]:
    """
    Return aggregated performance metrics by territory.
    """
    logger.info("GET /api/v1/reps/territory/summary")
    try:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT
                territory,
                COUNT(*)                            AS num_reps,
                SUM(total_sales)                    AS total_sales,
                ROUND(AVG(avg_gross_profit), 2)     AS avg_gross_profit,
                ROUND(AVG(avg_satisfaction_score), 2) AS avg_satisfaction,
                ROUND(AVG(quota_attainment_2yr), 4) AS avg_quota_attainment
            FROM main_marts.dim_reps
            GROUP BY territory
            ORDER BY total_sales DESC
        """).fetchall()
        conn.close()

        return [
            {
                "territory": r[0],
                "num_reps": r[1],
                "total_sales": r[2],
                "avg_gross_profit": r[3],
                "avg_satisfaction": r[4],
                "avg_quota_attainment": r[5],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"territory_summary error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ------------------------------------------------------------------
# All reps list
# ------------------------------------------------------------------


@router.get("/")
def list_reps() -> list[dict[str, Any]]:
    """Return all reps with key metrics."""
    logger.info("GET /api/v1/reps/")
    try:
        conn = _get_conn()
        rows = conn.execute("""
            SELECT
                rep_id,
                rep_name,
                territory,
                total_sales,
                avg_gross_profit,
                avg_satisfaction_score,
                quota_attainment_2yr
            FROM main_marts.dim_reps
            ORDER BY total_gross_profit DESC
        """).fetchall()
        conn.close()

        return [
            {
                "rep_id": r[0],
                "rep_name": r[1],
                "territory": r[2],
                "total_sales": r[3],
                "avg_gross_profit": r[4],
                "avg_satisfaction": r[5],
                "quota_attainment_2yr": r[6],
            }
            for r in rows
        ]
    except Exception as exc:
        logger.error(f"list_reps error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
