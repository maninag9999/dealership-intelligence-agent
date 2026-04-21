"""
Synthetic data generators with realistic statistical patterns.

Latent columns (prefixed _) encode the ground-truth data-generating process.
The pipeline strips them before writing public Parquet files.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta
from enum import StrEnum

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

from dealership.common.config import get_settings
from dealership.ingestion.synthetic.catalog import VEHICLE_CATALOG, popularity_weights

settings = get_settings()


class RepArchetype(StrEnum):
    HIGH_VOLUME = "high_volume"
    RELATIONSHIP = "relationship"
    STRUGGLING = "struggling"


# Keys are plain strings so rng.choice() lookups never fail
_ARCHETYPE_PARAMS: dict[str, dict] = {
    "high_volume": {
        "close_rate_mean": 0.34,
        "close_rate_std": 0.04,
        "margin_tendency_mean": -400.0,
        "margin_tendency_std": 200.0,
        "sat_bias": -0.15,
        "mix_weight": 0.25,
    },
    "relationship": {
        "close_rate_mean": 0.22,
        "close_rate_std": 0.03,
        "margin_tendency_mean": 900.0,
        "margin_tendency_std": 300.0,
        "sat_bias": 0.35,
        "mix_weight": 0.40,
    },
    "struggling": {
        "close_rate_mean": 0.11,
        "close_rate_std": 0.05,
        "margin_tendency_mean": -1400.0,
        "margin_tendency_std": 500.0,
        "sat_bias": -0.30,
        "mix_weight": 0.35,
    },
}

_INCOME_BRACKETS = ["<$35k", "$35k-$55k", "$55k-$80k", "$80k-$120k", "$120k+"]
_INCOME_WEIGHTS = [0.18, 0.25, 0.28, 0.20, 0.09]
_CREDIT_BY_INCOME: dict[str, tuple[float, float]] = {
    "<$35k": (620.0, 55.0),
    "$35k-$55k": (660.0, 50.0),
    "$55k-$80k": (700.0, 45.0),
    "$80k-$120k": (735.0, 40.0),
    "$120k+": (760.0, 35.0),
}
_CONDITIONS = ["New", "Certified Pre-Owned", "Used"]
_CONDITION_WEIGHTS = [0.38, 0.22, 0.40]
_CONDITION_AGE_RANGES: dict[str, tuple[float, float]] = {
    "New": (0.0, 0.5),
    "Certified Pre-Owned": (1.0, 4.0),
    "Used": (2.0, 12.0),
}
_CONDITION_MILEAGE: dict[str, tuple[float, float]] = {
    "New": (25.0, 20.0),
    "Certified Pre-Owned": (28000.0, 12000.0),
    "Used": (72000.0, 35000.0),
}
_FINANCING_TYPES = ["Cash", "Dealer Finance", "Bank Finance", "Credit Union", "Lease"]
_FINANCING_WEIGHTS_BY_INCOME: dict[str, list[float]] = {
    "<$35k": [0.05, 0.45, 0.30, 0.15, 0.05],
    "$35k-$55k": [0.10, 0.38, 0.28, 0.18, 0.06],
    "$55k-$80k": [0.18, 0.32, 0.24, 0.16, 0.10],
    "$80k-$120k": [0.28, 0.22, 0.20, 0.16, 0.14],
    "$120k+": [0.40, 0.10, 0.15, 0.12, 0.23],
}
_MONTH_DISCOUNT: dict[int, float] = {
    1: 1.00,
    2: 0.95,
    3: 0.90,
    4: 0.92,
    5: 0.93,
    6: 0.95,
    7: 0.97,
    8: 0.98,
    9: 0.95,
    10: 0.97,
    11: 1.05,
    12: 1.18,
}


def _new_id() -> str:
    return str(uuid.uuid4())


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def _random_date(rng: np.random.Generator, start: date, end: date) -> date:
    return start + timedelta(days=int(rng.integers(0, (end - start).days + 1)))


# ------------------------------------------------------------------
# 1. Sales Representatives
# ------------------------------------------------------------------


def generate_reps(
    n: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate sales reps with latent archetypes (high_volume/relationship/struggling)."""
    n = n or settings.synthetic_num_reps
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed)
    fake = Faker("en_US")
    Faker.seed(int(rng.integers(0, 2**31)))
    logger.info(f"Generating {n} sales reps ...")

    # Use plain string keys — no enum conversion needed after rng.choice()
    archetype_keys = list(_ARCHETYPE_PARAMS.keys())
    mix_weights = [_ARCHETYPE_PARAMS[k]["mix_weight"] for k in archetype_keys]
    territories = ["North", "South", "East", "West", "Central"]

    rows: list[dict] = []
    for _ in range(n):
        arch = str(rng.choice(archetype_keys, p=mix_weights))  # plain string, always valid key
        p = _ARCHETYPE_PARAMS[arch]
        hire = _random_date(rng, _parse_date("2015-01-01"), _parse_date("2022-06-30"))
        rows.append(
            {
                "rep_id": _new_id(),
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "hire_date": hire.isoformat(),
                "years_experience": round(float(rng.uniform(0.5, 15.0)), 1),
                "territory": str(rng.choice(territories)),
                "monthly_quota_usd": int(rng.integers(8, 25)) * 5_000,
                "_archetype": arch,
                "_base_close_rate": round(
                    float(np.clip(rng.normal(p["close_rate_mean"], p["close_rate_std"]), 0.05, 0.55)),
                    4,
                ),
                "_margin_tendency": round(float(rng.normal(p["margin_tendency_mean"], p["margin_tendency_std"])), 2),
            }
        )

    df = pd.DataFrame(rows)
    logger.success(f"Reps done — archetype mix: {df['_archetype'].value_counts().to_dict()}")
    return df


# ------------------------------------------------------------------
# 2. Customers
# ------------------------------------------------------------------


def generate_customers(
    n: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate customers with income/credit/price-sensitivity correlations."""
    n = n or settings.synthetic_num_customers
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed + 1)
    fake = Faker("en_US")
    Faker.seed(int(rng.integers(0, 2**31)))
    logger.info(f"Generating {n} customers ...")

    rows: list[dict] = []
    for _ in range(n):
        ib = str(rng.choice(_INCOME_BRACKETS, p=_INCOME_WEIGHTS))
        mu, sig = _CREDIT_BY_INCOME[ib]
        credit = int(np.clip(rng.normal(mu, sig), 300, 850))
        ps_mean = 0.75 - _INCOME_BRACKETS.index(ib) * 0.12
        ps = float(np.clip(rng.normal(ps_mean, 0.12), 0.0, 1.0))
        rows.append(
            {
                "customer_id": _new_id(),
                "first_name": fake.first_name(),
                "last_name": fake.last_name(),
                "age": int(np.clip(rng.normal(38.0, 12.0), 18, 80)),
                "income_bracket": ib,
                "credit_score": credit,
                "zip_code": fake.zipcode(),
                "state": fake.state_abbr(),
                "preferred_contact": str(rng.choice(["Email", "Phone", "Text", "In-Person"])),
                "is_returning": bool(rng.random() < 0.22),
                "_price_sensitivity": round(ps, 4),
            }
        )

    df = pd.DataFrame(rows)
    logger.success(f"Customers done — income mix: {df['income_bracket'].value_counts().to_dict()}")
    return df


# ------------------------------------------------------------------
# 3. Vehicles
# ------------------------------------------------------------------


def generate_vehicles(
    n: int | None = None,
    *,
    rng: np.random.Generator | None = None,
    reference_date: date | None = None,
) -> pd.DataFrame:
    """Generate vehicle inventory using catalog depreciation model."""
    n = n or settings.synthetic_num_vehicles
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed + 2)
    reference_date = reference_date or _parse_date(settings.synthetic_date_start)
    logger.info(f"Generating {n} vehicles ...")

    weights = popularity_weights()
    cat_idx = list(range(len(VEHICLE_CATALOG)))
    cond_mul = {"New": 1.00, "Certified Pre-Owned": 0.92, "Used": 0.84}

    rows: list[dict] = []
    for _ in range(n):
        spec = VEHICLE_CATALOG[int(rng.choice(cat_idx, p=weights))]
        cond = str(rng.choice(_CONDITIONS, p=_CONDITION_WEIGHTS))
        age = float(rng.uniform(*_CONDITION_AGE_RANGES[cond]))
        yr = reference_date.year - int(age)
        mil = max(0, int(rng.normal(*_CONDITION_MILEAGE[cond])))
        trim = str(rng.choice(spec.available_trims))
        color = str(rng.choice(spec.available_colors))
        ti = list(spec.available_trims).index(trim)
        tf = ti / max(1, len(spec.available_trims) - 1)
        msrp = int(spec.msrp_low + tf * (spec.msrp_high - spec.msrp_low))
        tb = spec.depreciated_value(age, mil)
        ask = max(3_500.0, round(tb * cond_mul[cond] * float(rng.normal(1.0, 0.04)), -2))
        cost = round(ask * float(rng.uniform(0.80, 0.93)), -2)
        arr = _random_date(
            rng,
            reference_date - timedelta(days=90),
            reference_date + timedelta(days=int(365 * 1.8)),
        )
        vc = "".join(
            str(rng.integers(0, 10)) if rng.random() < 0.5 else chr(int(rng.integers(65, 91))) for _ in range(10)
        )
        rows.append(
            {
                "vehicle_id": _new_id(),
                "vin": f"1{spec.make[:1].upper()}T{yr%100:02d}{vc}"[:17],
                "make": spec.make,
                "model": spec.model,
                "year": yr,
                "trim": trim,
                "color": color,
                "segment": spec.segment,
                "condition": cond,
                "mileage": mil,
                "msrp": msrp,
                "cost_basis": int(cost),
                "asking_price": int(ask),
                "arrived_date": arr.isoformat(),
                "_true_base_value": round(tb, 2),
            }
        )

    df = pd.DataFrame(rows)
    logger.success(f"Vehicles done — condition mix: {df['condition'].value_counts().to_dict()}")
    return df


# ------------------------------------------------------------------
# 4. Sales
# ------------------------------------------------------------------


def generate_sales(
    vehicles_df: pd.DataFrame,
    customers_df: pd.DataFrame,
    reps_df: pd.DataFrame,
    n: int | None = None,
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate sales with correlated days-on-lot, December spike, rep heterogeneity."""
    n = n or settings.synthetic_num_sales
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed + 3)
    logger.info(f"Generating {n} sales ...")

    ds, de = _parse_date(settings.synthetic_date_start), _parse_date(settings.synthetic_date_end)

    v_ids = vehicles_df["vehicle_id"].values
    va = dict(zip(vehicles_df["vehicle_id"], vehicles_df["asking_price"], strict=False))
    vc = dict(zip(vehicles_df["vehicle_id"], vehicles_df["cost_basis"], strict=False))
    vr = dict(zip(vehicles_df["vehicle_id"], vehicles_df["arrived_date"], strict=False))
    vt = dict(zip(vehicles_df["vehicle_id"], vehicles_df["_true_base_value"], strict=False))

    c_ids = customers_df["customer_id"].values
    ci = dict(zip(customers_df["customer_id"], customers_df["income_bracket"], strict=False))
    cp = dict(zip(customers_df["customer_id"], customers_df["_price_sensitivity"], strict=False))

    r_ids = reps_df["rep_id"].values
    ra = dict(zip(reps_df["rep_id"], reps_df["_archetype"], strict=False))
    rm = dict(zip(reps_df["rep_id"], reps_df["_margin_tendency"], strict=False))
    rsb = {rid: _ARCHETYPE_PARAMS[str(ra[rid])]["sat_bias"] for rid in r_ids}

    rows: list[dict] = []
    for _ in range(n):
        vid = str(rng.choice(v_ids))
        cid = str(rng.choice(c_ids))
        rid = str(rng.choice(r_ids))
        sd = _random_date(rng, ds, de)

        ask = float(va[vid])
        cost = float(vc[vid])
        tb = float(vt[vid])
        arr = _parse_date(str(vr[vid]))

        dol = int(np.clip(max(5, int(rng.normal(30, 12))) * ((ask / max(tb, 1.0)) ** 2.5), 3, 180))
        sd = max(sd, arr + timedelta(days=dol))
        if sd > de:
            sd = de

        ps = float(cp[cid])
        mm = _MONTH_DISCOUNT[sd.month]
        base_dp = float(np.clip(rng.normal(0.04, 0.015) * ps, 0.0, 0.15))
        seasonal_bonus = max(0.0, (mm - 1.0) * 0.12)  # Dec: +2.2%, Mar: 0%
        dp = float(np.clip(base_dp + seasonal_bonus, 0.0, 0.18))
        sp = max(cost * 1.01, round(ask - round(ask * dp, 2), -1))
        da = round(ask - sp, 2)
        dp2 = round(da / ask, 4) if ask > 0 else 0.0

        arch = str(ra[rid])
        ns = 800.0 if arch == "struggling" else 300.0
        gp = round((sp - cost) + float(rm[rid]) + float(rng.normal(0, ns)), 2)

        tv = int(rng.integers(1_000, 22_000)) if rng.random() < 0.38 else 0
        fw = _FINANCING_WEIGHTS_BY_INCOME.get(str(ci[cid]), [0.2] * 5)
        ft = str(rng.choice(_FINANCING_TYPES, p=fw))
        sat = float(np.clip(round(rng.normal(3.6 + float(rsb[rid]), 0.7) * 2) / 2, 1.0, 5.0))

        rows.append(
            {
                "sale_id": _new_id(),
                "vehicle_id": vid,
                "customer_id": cid,
                "rep_id": rid,
                "sale_date": sd.isoformat(),
                "asking_price_at_sale": int(ask),
                "sale_price": int(sp),
                "discount_amount": int(da),
                "discount_pct": dp2,
                "days_on_lot": dol,
                "trade_in_value": tv,
                "financing_type": ft,
                "gross_profit": int(gp),
                "customer_satisfaction_score": sat,
                "_rep_archetype": arch,
            }
        )

    df = pd.DataFrame(rows)
    logger.success(
        f"Sales done — avg discount {df['discount_pct'].mean():.1%}, "
        f"avg gross ${df['gross_profit'].mean():,.0f}, "
        f"avg days-on-lot {df['days_on_lot'].mean():.1f}"
    )
    return df
