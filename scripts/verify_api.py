#!/usr/bin/env python
"""
Verification script for the Dealership Intelligence Agent API.

Starts the API server, tests all endpoints, then shuts down.

Usage::

    # Start API first in a separate terminal:
    uv run python scripts/start_api.py --skip-chroma

    # Then run verify:
    uv run python scripts/verify_api.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import typer
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

app = typer.Typer(add_completion=False)


def _check(name: str, ok: bool, detail: str) -> bool:
    icon = "OK" if ok else "XX"
    (logger.success if ok else logger.error)(f"[{icon}] {name} -> {detail}")
    return ok


@app.command()
def main(
    base_url: str = typer.Option(
        "http://localhost:8000",
        "--base-url",
        help="Base URL of the API",
    ),
    timeout: float = typer.Option(
        10.0,
        "--timeout",
        help="Request timeout in seconds",
    ),
) -> None:
    """Verify all API endpoints are working correctly."""
    logger.remove()
    fmt = "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}"
    logger.add(sys.stderr, format=fmt, colorize=True, level="DEBUG")

    logger.info(f"Verifying API at {base_url}")
    logger.info("-" * 50)

    results = []
    client = httpx.Client(base_url=base_url, timeout=timeout)

    # ── 1. Root ──────────────────────────────────────────────────
    try:
        r = client.get("/")
        results.append(
            _check(
                "GET /",
                r.status_code == 200,
                f"status={r.status_code}",
            )
        )
    except Exception as exc:
        results.append(_check("GET /", False, f"connection failed: {exc}"))
        logger.error("API is not running — start it first with:")
        logger.error("  uv run python scripts/start_api.py --skip-chroma")
        raise typer.Exit(code=1) from None

    # ── 2. Health ─────────────────────────────────────────────────
    try:
        r = client.get("/health")
        data = r.json()
        results.append(
            _check(
                "GET /health",
                r.status_code == 200 and data.get("status") == "ok",
                f"warehouse={data.get('warehouse')} counts={data.get('counts')}",
            )
        )
    except Exception as exc:
        results.append(_check("GET /health", False, str(exc)))

    # ── 3. Sales summary ──────────────────────────────────────────
    try:
        r = client.get("/api/v1/sales/summary")
        data = r.json()
        ok = r.status_code == 200 and data.get("total_sales", 0) > 0
        results.append(
            _check(
                "GET /api/v1/sales/summary",
                ok,
                f"total_sales={data.get('total_sales')} avg_gross=${data.get('avg_gross_profit')}",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/sales/summary", False, str(exc)))

    # ── 4. Sales monthly ──────────────────────────────────────────
    try:
        r = client.get("/api/v1/sales/monthly")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/sales/monthly",
                r.status_code == 200 and len(data) > 0,
                f"{len(data)} months returned",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/sales/monthly", False, str(exc)))

    # ── 5. Top vehicles ───────────────────────────────────────────
    try:
        r = client.get("/api/v1/sales/top-vehicles?limit=5")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/sales/top-vehicles",
                r.status_code == 200 and len(data) == 5,
                f"top vehicle: {data[0].get('make')} {data[0].get('model')} ({data[0].get('total_sales')} sales)",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/sales/top-vehicles", False, str(exc)))

    # ── 6. Sales aging ────────────────────────────────────────────
    try:
        r = client.get("/api/v1/sales/aging")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/sales/aging",
                r.status_code == 200 and len(data) > 0,
                f"{len(data)} aging buckets",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/sales/aging", False, str(exc)))

    # ── 7. Recent sales ───────────────────────────────────────────
    try:
        r = client.get("/api/v1/sales/recent?limit=5")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/sales/recent",
                r.status_code == 200 and len(data) == 5,
                f"latest sale: {data[0].get('make')} {data[0].get('model')}",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/sales/recent", False, str(exc)))

    # ── 8. Reps list ──────────────────────────────────────────────
    try:
        r = client.get("/api/v1/reps/")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/reps/",
                r.status_code == 200 and len(data) > 0,
                f"{len(data)} reps returned",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/reps/", False, str(exc)))

    # ── 9. Reps leaderboard ───────────────────────────────────────
    try:
        r = client.get("/api/v1/reps/leaderboard?limit=5")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/reps/leaderboard",
                r.status_code == 200 and len(data) == 5,
                f"top rep: {data[0].get('rep_name')} gross=${data[0].get('avg_gross_profit')}",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/reps/leaderboard", False, str(exc)))

    # ── 10. Territory summary ─────────────────────────────────────
    try:
        r = client.get("/api/v1/reps/territory/summary")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/reps/territory/summary",
                r.status_code == 200 and len(data) > 0,
                f"{len(data)} territories",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/reps/territory/summary", False, str(exc)))

    # ── 11. LLM providers ─────────────────────────────────────────
    try:
        r = client.get("/api/v1/llm/providers")
        data = r.json()
        results.append(
            _check(
                "GET /api/v1/llm/providers",
                r.status_code == 200,
                f"active_provider={data.get('active_provider')}",
            )
        )
    except Exception as exc:
        results.append(_check("GET /api/v1/llm/providers", False, str(exc)))

    # ── 12. LLM ask ───────────────────────────────────────────────
    try:
        r = client.post(
            "/api/v1/llm/ask",
            json={"prompt": "What is the most important metric for a dealership?"},
            timeout=30.0,
        )
        data = r.json()
        ok = r.status_code == 200 and len(data.get("response", "")) > 10
        results.append(
            _check(
                "POST /api/v1/llm/ask",
                ok,
                f"provider={data.get('provider')} response_len={len(data.get('response',''))}",
            )
        )
    except Exception as exc:
        results.append(_check("POST /api/v1/llm/ask", False, str(exc)))

    client.close()

    # ── Summary ───────────────────────────────────────────────────
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
