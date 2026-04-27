"""
LLM routes for the Dealership Intelligence Agent API.

Tries providers in this order:
1. Groq  — free, runs anywhere, ultra fast
2. Ollama — local, no internet needed
3. Mock   — always works, never crashes

This means the API works for recruiters without any LLM setup.
"""

from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException
from groq import Groq
from loguru import logger
from pydantic import BaseModel

from dealership.common.config import get_settings

settings = get_settings()
router = APIRouter()

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

_SYSTEM_PROMPT = """You are an expert automotive dealership analyst AI assistant.
You have deep knowledge of vehicle sales, inventory management, rep performance,
and dealership operations. You provide concise, data-driven insights.
Always be specific and actionable in your responses.
Keep responses under 200 words unless asked for detail."""


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------


class LLMRequest(BaseModel):
    """Request body for LLM endpoint."""

    prompt: str
    system: str = _SYSTEM_PROMPT
    max_tokens: int = 512


class LLMResponse(BaseModel):
    """Response from LLM endpoint."""

    response: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ------------------------------------------------------------------
# Provider functions
# ------------------------------------------------------------------


def _try_groq(prompt: str, system: str, max_tokens: int) -> LLMResponse | None:
    """
    Try Groq API.

    Returns None if key not configured or request fails.
    """
    api_key = settings.groq_api_key.get_secret_value()
    if not api_key or api_key.strip() == "":
        logger.debug("Groq API key not configured — skipping")
        return None

    try:
        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        text = completion.choices[0].message.content or ""
        logger.success(f"Groq response received ({len(text)} chars)")
        return LLMResponse(
            response=text,
            provider="groq",
            model=settings.groq_model,
            prompt_tokens=completion.usage.prompt_tokens,
            completion_tokens=completion.usage.completion_tokens,
        )
    except Exception as exc:
        logger.warning(f"Groq failed: {exc}")
        return None


def _try_ollama(prompt: str, system: str, max_tokens: int) -> LLMResponse | None:
    """
    Try local Ollama instance.

    Returns None if Ollama is not running or request fails.
    """
    try:
        url = f"{settings.ollama_base_url}/api/chat"
        payload = {
            "model": settings.ollama_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data["message"]["content"]
            logger.success(f"Ollama response received ({len(text)} chars)")
            return LLMResponse(
                response=text,
                provider="ollama",
                model=settings.ollama_model,
            )
    except Exception as exc:
        logger.warning(f"Ollama failed: {exc}")
        return None


def _mock_response(prompt: str) -> LLMResponse:
    """
    Always-available mock response.

    Used when both Groq and Ollama are unavailable.
    Ensures the API never crashes for demo purposes.
    """
    logger.warning("Using mock LLM response — configure Groq or Ollama for real responses")
    return LLMResponse(
        response=(
            f"[Mock Response] I received your query: '{prompt[:100]}'. "
            "To enable real AI responses, either: "
            "(1) Add GROQ_API_KEY to your .env file (free at console.groq.com), or "
            "(2) Install Ollama from ollama.com and run 'ollama pull llama3'."
        ),
        provider="mock",
        model="mock",
    )


def _call_llm(prompt: str, system: str, max_tokens: int) -> LLMResponse:
    """
    Call LLM with automatic provider fallback.

    Order: Groq → Ollama → Mock
    """
    return _try_groq(prompt, system, max_tokens) or _try_ollama(prompt, system, max_tokens) or _mock_response(prompt)


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------


@router.post("/ask")
def ask(request: LLMRequest) -> LLMResponse:
    """
    Ask the dealership AI assistant a question.

    Automatically uses Groq if configured, otherwise Ollama,
    otherwise returns a helpful mock response.

    Example prompts
    ---------------
    - "Which vehicle segments have the highest gross profit?"
    - "What does a days-on-lot of 45 indicate about pricing?"
    - "How should a struggling rep improve their close rate?"
    """
    logger.info(f"POST /api/v1/llm/ask — prompt={request.prompt[:60]}...")
    try:
        return _call_llm(request.prompt, request.system, request.max_tokens)
    except Exception as exc:
        logger.error(f"LLM ask error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/analyze/vehicle/{vehicle_id}")
def analyze_vehicle(vehicle_id: str) -> LLMResponse:
    """
    Generate AI analysis for a specific vehicle.

    Fetches vehicle data from DuckDB and asks the LLM
    to provide pricing and inventory recommendations.
    """
    import duckdb

    logger.info(f"GET /api/v1/llm/analyze/vehicle/{vehicle_id}")

    try:
        conn = duckdb.connect(str(settings.duckdb_file()), read_only=True)
        row = conn.execute(
            """
            SELECT make, model, year, trim, condition,
                   mileage, asking_price, msrp, arrived_date
            FROM raw.vehicles
            WHERE vehicle_id = ?
        """,
            [vehicle_id],
        ).fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")

        prompt = (
            f"Analyze this vehicle listing and provide recommendations:\n"
            f"Vehicle: {row[2]} {row[0]} {row[1]} {row[3]}\n"
            f"Condition: {row[4]}, Mileage: {row[5]:,}\n"
            f"Asking Price: ${row[6]:,}, MSRP: ${row[7]:,}\n"
            f"Arrived: {row[8]}\n\n"
            f"Is the pricing competitive? What should the rep focus on to close this deal?"
        )
        return _call_llm(prompt, _SYSTEM_PROMPT, 512)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"analyze_vehicle error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/analyze/rep/{rep_id}")
def analyze_rep(rep_id: str) -> LLMResponse:
    """
    Generate AI coaching analysis for a specific sales rep.

    Fetches rep performance from dim_reps and generates
    personalised coaching recommendations.
    """
    import duckdb

    logger.info(f"GET /api/v1/llm/analyze/rep/{rep_id}")

    try:
        conn = duckdb.connect(str(settings.duckdb_file()), read_only=True)
        row = conn.execute(
            """
            SELECT rep_name, territory, total_sales, avg_gross_profit,
                   avg_discount_pct, avg_days_on_lot, avg_satisfaction_score,
                   quota_attainment_2yr, loss_sales, stale_sales
            FROM main_marts.dim_reps
            WHERE rep_id = ?
        """,
            [rep_id],
        ).fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"Rep {rep_id} not found")

        prompt = (
            f"Provide coaching recommendations for this sales rep:\n"
            f"Rep: {row[0]} ({row[1]} territory)\n"
            f"Total Sales: {row[2]}, Avg Gross Profit: ${row[3]:,.0f}\n"
            f"Avg Discount: {row[4]:.1%}, Avg Days on Lot: {row[5]:.0f}\n"
            f"Satisfaction Score: {row[6]}/5\n"
            f"Quota Attainment: {row[7]:.1%}\n"
            f"Loss Sales: {row[8]}, Stale Inventory Sold: {row[9]}\n\n"
            f"What are their strengths? Where should they improve? "
            f"Give 3 specific actionable recommendations."
        )
        return _call_llm(prompt, _SYSTEM_PROMPT, 512)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"analyze_rep error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/providers")
def list_providers() -> dict:
    """
    Return which LLM providers are currently configured.

    Useful for debugging and UI display.
    """
    groq_key = settings.groq_api_key.get_secret_value()
    groq_configured = bool(groq_key and groq_key.strip())

    ollama_available = False
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_available = resp.status_code == 200
    except Exception:
        pass

    return {
        "groq": {
            "configured": groq_configured,
            "model": settings.groq_model,
        },
        "ollama": {
            "available": ollama_available,
            "model": settings.ollama_model,
            "url": settings.ollama_base_url,
        },
        "active_provider": ("groq" if groq_configured else "ollama" if ollama_available else "mock"),
    }
