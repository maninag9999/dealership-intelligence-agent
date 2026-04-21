"""
Bootstrap — writes all Day 1 Part 2 source files to the correct locations.
Run once from the project root:
    uv run python bootstrap_day1_part2.py
"""

from pathlib import Path

ROOT = Path(__file__).parent
files: dict[str, str] = {}

# ======================================================================
# src/dealership/common/config.py
# ======================================================================
files["src/dealership/common/config.py"] = '''"""
Runtime configuration for the Dealership Intelligence Agent.

All settings are loaded from environment variables / .env file via
pydantic-settings.  Sensitive values (passwords, API keys) are typed as
``SecretStr`` so they are redacted in repr() / logging output.

Usage::

    from dealership.common.config import get_settings

    settings = get_settings()
    db_url = settings.postgres_dsn()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings loaded from .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "dealership"
    postgres_user: str = "dealership_user"
    postgres_password: SecretStr = SecretStr("changeme")

    # DuckDB
    duckdb_path: str = "data/warehouse/dealership.duckdb"

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "dealership_docs"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    # Synthetic data
    synthetic_seed: int = 42
    synthetic_num_reps: int = 20
    synthetic_num_customers: int = 800
    synthetic_num_vehicles: int = 450
    synthetic_num_sales: int = 1_400
    synthetic_date_start: str = "2022-01-01"
    synthetic_date_end: str = "2023-12-31"

    # Paths
    data_raw_path: str = "data/raw/synthetic"
    data_warehouse_path: str = "data/warehouse"
    data_debug_path: str = "data/debug/synthetic"

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/dealership.log"

    @field_validator("postgres_port", "chroma_port", mode="before")
    @classmethod
    def _port_range(cls, v: int) -> int:
        if not (1 <= int(v) <= 65535):
            raise ValueError(f"Port must be 1-65535, got {v}")
        return int(v)

    @field_validator("log_level", mode="before")
    @classmethod
    def _log_level_upper(cls, v: str) -> str:
        v = v.upper()
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v

    @model_validator(mode="after")
    def _ensure_paths_exist(self) -> "Settings":
        for attr in ("data_raw_path", "data_warehouse_path", "data_debug_path"):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
        return self

    def postgres_dsn(self, *, driver: str = "postgresql+psycopg2") -> str:
        """Return SQLAlchemy-compatible DSN (unwraps secret)."""
        pwd = self.postgres_password.get_secret_value()
        return (
            f"{driver}://{self.postgres_user}:{pwd}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def duckdb_file(self) -> Path:
        return Path(self.duckdb_path)

    def raw_path(self) -> Path:
        return Path(self.data_raw_path)

    def debug_path(self) -> Path:
        return Path(self.data_debug_path)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached Settings singleton."""
    return Settings()
'''

# ======================================================================
# src/dealership/ingestion/synthetic/catalog.py
# ======================================================================
files["src/dealership/ingestion/synthetic/catalog.py"] = '''"""
Static vehicle catalog — 28 models across 7 brands.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass(frozen=True)
class VehicleSpec:
    make: str
    model: str
    segment: str
    msrp_low: int
    msrp_high: int
    annual_depreciation_rate: float
    popularity_weight: float
    reliability_score: float
    avg_days_on_lot: int
    typical_buyer_age_mean: float
    typical_buyer_age_std: float
    available_colors: tuple[str, ...] = field(
        default=("White", "Black", "Silver", "Gray", "Blue", "Red")
    )
    available_trims: tuple[str, ...] = field(
        default=("Base", "Mid", "Premium")
    )

    def msrp_midpoint(self) -> float:
        return (self.msrp_low + self.msrp_high) / 2.0

    def depreciated_value(self, age_years: float, mileage: int) -> float:
        base = self.msrp_midpoint()
        age_factor = (1.0 - self.annual_depreciation_rate) ** age_years
        expected_miles = age_years * 12_000
        excess_miles = max(0, mileage - expected_miles)
        mileage_penalty = excess_miles * 0.08
        return max(base * age_factor - mileage_penalty, base * 0.10)


VEHICLE_CATALOG: list[VehicleSpec] = [
    # TOYOTA
    VehicleSpec("Toyota","Camry","Sedan",26420,35720,0.155,1.40,0.89,22,42.0,11.0,
        ("White","Black","Silver","Gray","Red","Blue"),("LE","SE","XSE","XLE")),
    VehicleSpec("Toyota","RAV4","SUV",28475,40760,0.145,1.65,0.91,18,38.0,10.0,
        ("White","Black","Gray","Blue","Red","Green"),("LE","XLE","Adventure","Limited","TRD Off-Road")),
    VehicleSpec("Toyota","Tacoma","Truck",28700,44910,0.110,1.35,0.87,15,35.0,9.0,
        ("White","Black","Silver","Gray","Red","Army Green"),("SR","SR5","TRD Sport","TRD Off-Road","Limited")),
    VehicleSpec("Toyota","Highlander","SUV",36420,52875,0.160,1.10,0.88,24,44.0,9.5),
    # HONDA
    VehicleSpec("Honda","Accord","Sedan",27295,38640,0.160,1.25,0.88,25,40.0,12.0,
        ("White","Black","Silver","Gray","Blue","Red"),("LX","Sport","EX-L","Touring")),
    VehicleSpec("Honda","CR-V","SUV",29400,39200,0.150,1.45,0.90,20,37.0,10.0,
        ("White","Black","Silver","Gray","Blue","Red"),("LX","EX","EX-L","Sport","Touring")),
    VehicleSpec("Honda","Pilot","SUV",38050,51200,0.165,0.90,0.85,28,43.0,8.5,
        ("White","Black","Silver","Gray","Blue","Red"),("Sport","EX-L","TrailSport","Elite")),
    VehicleSpec("Honda","Civic","Sedan",23950,31900,0.170,1.20,0.87,21,29.0,8.0,
        ("White","Black","Silver","Blue","Red","Sonic Gray"),("LX","Sport","EX","Touring")),
    # FORD
    VehicleSpec("Ford","F-150","Truck",32145,79840,0.175,1.80,0.78,16,41.0,11.0,
        ("White","Black","Silver","Gray","Blue","Red","Rapid Red"),
        ("Regular","XL","XLT","Lariat","Platinum","Limited","Raptor")),
    VehicleSpec("Ford","Explorer","SUV",36760,56600,0.190,1.10,0.74,30,43.0,9.0,
        ("White","Black","Silver","Gray","Blue","Red"),("Base","XLT","ST","Limited","Platinum")),
    VehicleSpec("Ford","Mustang","Coupe",30920,59900,0.200,0.70,0.73,38,34.0,13.0,
        ("White","Black","Gray","Red","Blue","Yellow","Orange"),
        ("EcoBoost","GT","Mach 1","Shelby GT500")),
    VehicleSpec("Ford","Bronco","SUV",33695,65390,0.125,0.85,0.76,19,36.0,9.5,
        ("White","Black","Gray","Blue","Red","Green"),
        ("Base","Big Bend","Black Diamond","Outer Banks","Badlands","Wildtrak","Raptor")),
    # CHEVROLET
    VehicleSpec("Chevrolet","Silverado 1500","Truck",34600,72000,0.180,1.55,0.77,18,42.0,11.5,
        ("White","Black","Silver","Gray","Blue","Red","Green"),
        ("Work Truck","Custom","LT","RST","LTZ","High Country","ZR2")),
    VehicleSpec("Chevrolet","Equinox","SUV",26600,37100,0.185,1.15,0.80,26,38.0,11.0,
        ("White","Black","Silver","Gray","Blue","Red"),("LS","LT","RS","Premier")),
    VehicleSpec("Chevrolet","Traverse","SUV",35400,53200,0.195,0.85,0.79,29,44.0,8.5,
        ("White","Black","Silver","Gray","Blue","Red"),("LS","LT","RS","Premier","High Country")),
    VehicleSpec("Chevrolet","Colorado","Truck",27200,46500,0.160,0.80,0.81,22,37.0,10.0,
        ("White","Black","Silver","Gray","Blue","Red"),("WT","LT","Z71","ZR2")),
    # NISSAN
    VehicleSpec("Nissan","Rogue","SUV",27860,38510,0.195,1.20,0.79,26,38.0,10.5,
        ("White","Black","Silver","Gray","Blue","Red"),("S","SV","SL","Platinum")),
    VehicleSpec("Nissan","Frontier","Truck",29940,42680,0.150,0.75,0.82,24,39.0,10.0,
        ("White","Black","Silver","Gray","Blue","Red"),("S","SV","Pro-4X","PRO-X","SL")),
    VehicleSpec("Nissan","Altima","Sedan",24300,34250,0.200,0.95,0.77,30,36.0,12.0,
        ("White","Black","Silver","Gray","Blue","Red"),("S","SV","SR","SL","Platinum")),
    VehicleSpec("Nissan","Pathfinder","SUV",34175,51925,0.185,0.70,0.76,31,45.0,9.0),
    # HYUNDAI
    VehicleSpec("Hyundai","Tucson","SUV",27050,38950,0.185,1.00,0.83,27,37.0,10.5,
        ("White","Black","Silver","Gray","Blue","Red","Green"),("SE","SEL","N Line","Limited")),
    VehicleSpec("Hyundai","Santa Fe","SUV",32600,45100,0.180,0.85,0.84,28,42.0,9.0,
        ("White","Black","Silver","Gray","Blue","Red"),("SE","SEL","XRT","Limited","Calligraphy")),
    VehicleSpec("Hyundai","Elantra","Sedan",21950,30650,0.190,0.90,0.85,24,30.0,9.0,
        ("White","Black","Silver","Gray","Blue","Red","Yellow"),("SE","SEL","N Line","Limited")),
    VehicleSpec("Hyundai","Palisade","SUV",35850,52300,0.175,0.75,0.86,25,45.0,8.5,
        ("White","Black","Silver","Gray","Blue","Red"),("SE","SEL","XRT","Limited","Calligraphy")),
    # JEEP
    VehicleSpec("Jeep","Grand Cherokee","SUV",38995,74995,0.200,1.05,0.70,32,42.0,10.0,
        ("White","Black","Gray","Blue","Red","Green","Hydro Blue"),
        ("Laredo","Altitude","Limited","Overland","Summit","Trailhawk","4xe")),
    VehicleSpec("Jeep","Wrangler","SUV",31195,58300,0.130,1.10,0.68,20,35.0,11.0,
        ("White","Black","Gray","Blue","Red","Green","Orange","Yellow"),
        ("Sport","Sport S","Willys","Sahara","Rubicon","392")),
    VehicleSpec("Jeep","Compass","SUV",24695,35090,0.210,0.65,0.69,36,33.0,10.0,
        ("White","Black","Gray","Blue","Red","Green"),
        ("Sport","Latitude","Longitude","Limited","Trailhawk")),
    VehicleSpec("Jeep","Gladiator","Truck",37995,57500,0.155,0.60,0.71,34,38.0,10.5,
        ("White","Black","Gray","Blue","Red","Green"),
        ("Sport","Sport S","Willys","Overland","Rubicon","Mojave")),
]


def popularity_weights() -> list[float]:
    raw = [s.popularity_weight for s in VEHICLE_CATALOG]
    total = sum(raw)
    return [w / total for w in raw]
'''

# ======================================================================
# src/dealership/ingestion/synthetic/generators.py
# ======================================================================
files["src/dealership/ingestion/synthetic/generators.py"] = '''"""
Synthetic data generators with realistic statistical patterns.

Latent columns (prefixed _) encode the ground-truth data-generating process.
The pipeline strips them before writing public Parquet files.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta
from enum import Enum

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger

from dealership.common.config import get_settings
from dealership.ingestion.synthetic.catalog import VEHICLE_CATALOG, popularity_weights

settings = get_settings()


class RepArchetype(str, Enum):
    HIGH_VOLUME = "high_volume"
    RELATIONSHIP = "relationship"
    STRUGGLING = "struggling"


_ARCHETYPE_PARAMS: dict[RepArchetype, dict] = {
    RepArchetype.HIGH_VOLUME:  {"close_rate_mean":0.34,"close_rate_std":0.04,
        "margin_tendency_mean":-400.0,"margin_tendency_std":200.0,"sat_bias":-0.15,"mix_weight":0.25},
    RepArchetype.RELATIONSHIP: {"close_rate_mean":0.22,"close_rate_std":0.03,
        "margin_tendency_mean":900.0,"margin_tendency_std":300.0,"sat_bias":+0.35,"mix_weight":0.40},
    RepArchetype.STRUGGLING:   {"close_rate_mean":0.11,"close_rate_std":0.05,
        "margin_tendency_mean":-1400.0,"margin_tendency_std":500.0,"sat_bias":-0.30,"mix_weight":0.35},
}

_INCOME_BRACKETS = ["<$35k","$35k-$55k","$55k-$80k","$80k-$120k","$120k+"]
_INCOME_WEIGHTS  = [0.18, 0.25, 0.28, 0.20, 0.09]
_CREDIT_BY_INCOME = {
    "<$35k":(620.0,55.0),"$35k-$55k":(660.0,50.0),"$55k-$80k":(700.0,45.0),
    "$80k-$120k":(735.0,40.0),"$120k+":(760.0,35.0),
}
_CONDITIONS        = ["New","Certified Pre-Owned","Used"]
_CONDITION_WEIGHTS = [0.38, 0.22, 0.40]
_CONDITION_AGE_RANGES  = {"New":(0.0,0.5),"Certified Pre-Owned":(1.0,4.0),"Used":(2.0,12.0)}
_CONDITION_MILEAGE     = {"New":(25.0,20.0),"Certified Pre-Owned":(28000.0,12000.0),"Used":(72000.0,35000.0)}
_FINANCING_TYPES   = ["Cash","Dealer Finance","Bank Finance","Credit Union","Lease"]
_FINANCING_WEIGHTS_BY_INCOME = {
    "<$35k":      [0.05,0.45,0.30,0.15,0.05],
    "$35k-$55k":  [0.10,0.38,0.28,0.18,0.06],
    "$55k-$80k":  [0.18,0.32,0.24,0.16,0.10],
    "$80k-$120k": [0.28,0.22,0.20,0.16,0.14],
    "$120k+":     [0.40,0.10,0.15,0.12,0.23],
}
_MONTH_DISCOUNT_MULTIPLIERS = {
    1:1.00,2:0.95,3:0.90,4:0.92,5:0.93,6:0.95,
    7:0.97,8:0.98,9:0.95,10:0.97,11:1.05,12:1.18,
}


def _new_id() -> str:
    return str(uuid.uuid4())


def _random_date(rng: np.random.Generator, start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta + 1)))


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def generate_reps(n: int | None = None, *, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate sales representatives with latent archetypes."""
    n = n or settings.synthetic_num_reps
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed)
    fake = Faker("en_US")
    Faker.seed(int(rng.integers(0, 2**31)))
    logger.info(f"Generating {n} sales reps ...")

    archetypes = list(_ARCHETYPE_PARAMS.keys())
    mix_weights = [_ARCHETYPE_PARAMS[a]["mix_weight"] for a in archetypes]
    territories = ["North","South","East","West","Central"]
    rows = []

    for _ in range(n):
        archetype = rng.choice(archetypes, p=mix_weights)
        params = _ARCHETYPE_PARAMS[archetype]
        hire_date = _random_date(rng, _parse_date("2015-01-01"), _parse_date("2022-06-30"))
        close_rate = float(np.clip(rng.normal(params["close_rate_mean"], params["close_rate_std"]), 0.05, 0.55))
        margin_tendency = float(rng.normal(params["margin_tendency_mean"], params["margin_tendency_std"]))
        rows.append({
            "rep_id": _new_id(),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "hire_date": hire_date.isoformat(),
            "years_experience": round(float(rng.uniform(0.5, 15.0)), 1),
            "territory": str(rng.choice(territories)),
            "monthly_quota_usd": int(rng.integers(8, 25)) * 5_000,
            "_archetype": archetype.value,
            "_base_close_rate": round(close_rate, 4),
            "_margin_tendency": round(margin_tendency, 2),
        })

    df = pd.DataFrame(rows)
    logger.success(f"Reps done — mix: {df['_archetype'].value_counts().to_dict()}")
    return df


def generate_customers(n: int | None = None, *, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate customers with income/credit/price-sensitivity correlations."""
    n = n or settings.synthetic_num_customers
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed + 1)
    fake = Faker("en_US")
    Faker.seed(int(rng.integers(0, 2**31)))
    logger.info(f"Generating {n} customers ...")

    rows = []
    for _ in range(n):
        income_bracket = str(rng.choice(_INCOME_BRACKETS, p=_INCOME_WEIGHTS))
        credit_mu, credit_sigma = _CREDIT_BY_INCOME[income_bracket]
        credit_score = int(np.clip(rng.normal(credit_mu, credit_sigma), 300, 850))
        income_idx = _INCOME_BRACKETS.index(income_bracket)
        ps_mean = 0.75 - income_idx * 0.12
        price_sensitivity = float(np.clip(rng.normal(ps_mean, 0.12), 0.0, 1.0))
        rows.append({
            "customer_id": _new_id(),
            "first_name": fake.first_name(),
            "last_name": fake.last_name(),
            "age": int(np.clip(rng.normal(38.0, 12.0), 18, 80)),
            "income_bracket": income_bracket,
            "credit_score": credit_score,
            "zip_code": fake.zipcode(),
            "state": fake.state_abbr(),
            "preferred_contact": str(rng.choice(["Email","Phone","Text","In-Person"])),
            "is_returning": bool(rng.random() < 0.22),
            "_price_sensitivity": round(price_sensitivity, 4),
        })

    df = pd.DataFrame(rows)
    logger.success(f"Customers done — income mix: {df['income_bracket'].value_counts().to_dict()}")
    return df


def generate_vehicles(n: int | None = None, *, rng: np.random.Generator | None = None,
                      reference_date: date | None = None) -> pd.DataFrame:
    """Generate vehicle inventory using catalog depreciation model."""
    n = n or settings.synthetic_num_vehicles
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed + 2)
    reference_date = reference_date or _parse_date(settings.synthetic_date_start)
    logger.info(f"Generating {n} vehicles ...")

    weights = popularity_weights()
    catalog_idx = list(range(len(VEHICLE_CATALOG)))
    condition_multiplier = {"New":1.00,"Certified Pre-Owned":0.92,"Used":0.84}
    rows = []

    for _ in range(n):
        spec = VEHICLE_CATALOG[int(rng.choice(catalog_idx, p=weights))]
        condition = str(rng.choice(_CONDITIONS, p=_CONDITION_WEIGHTS))
        age_lo, age_hi = _CONDITION_AGE_RANGES[condition]
        age_years = float(rng.uniform(age_lo, age_hi))
        model_year = reference_date.year - int(age_years)
        mileage_mu, mileage_sigma = _CONDITION_MILEAGE[condition]
        mileage = max(0, int(rng.normal(mileage_mu, mileage_sigma)))
        trim = str(rng.choice(spec.available_trims))
        color = str(rng.choice(spec.available_colors))
        trim_idx = list(spec.available_trims).index(trim)
        trim_fraction = trim_idx / max(1, len(spec.available_trims) - 1)
        msrp = int(spec.msrp_low + trim_fraction * (spec.msrp_high - spec.msrp_low))
        true_base = spec.depreciated_value(age_years, mileage)
        noise = float(rng.normal(1.0, 0.04))
        asking = round(true_base * condition_multiplier[condition] * noise, -2)
        asking = max(asking, 3_500.0)
        cost_basis = round(asking * float(rng.uniform(0.80, 0.93)), -2)
        arrived_date = _random_date(rng,
            reference_date - timedelta(days=90),
            reference_date + timedelta(days=int(365 * 1.8)))
        vin_chars = "".join(
            str(rng.integers(0,10)) if rng.random() < 0.5 else chr(int(rng.integers(65,91)))
            for _ in range(10))
        vin = f"1{spec.make[:1].upper()}T{model_year%100:02d}{vin_chars}"[:17]
        rows.append({
            "vehicle_id": _new_id(), "vin": vin,
            "make": spec.make, "model": spec.model, "year": model_year,
            "trim": trim, "color": color, "segment": spec.segment,
            "condition": condition, "mileage": mileage, "msrp": msrp,
            "cost_basis": int(cost_basis), "asking_price": int(asking),
            "arrived_date": arrived_date.isoformat(),
            "_true_base_value": round(true_base, 2),
        })

    df = pd.DataFrame(rows)
    logger.success(f"Vehicles done — condition mix: {df['condition'].value_counts().to_dict()}")
    return df


def generate_sales(vehicles_df: pd.DataFrame, customers_df: pd.DataFrame,
                   reps_df: pd.DataFrame, n: int | None = None,
                   *, rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Generate sales with correlated days-on-lot, December spike, rep heterogeneity."""
    n = n or settings.synthetic_num_sales
    if rng is None:
        rng = np.random.default_rng(settings.synthetic_seed + 3)
    logger.info(f"Generating {n} sales transactions ...")

    date_start = _parse_date(settings.synthetic_date_start)
    date_end   = _parse_date(settings.synthetic_date_end)

    v_ids    = vehicles_df["vehicle_id"].values
    v_asking = dict(zip(vehicles_df["vehicle_id"], vehicles_df["asking_price"]))
    v_cost   = dict(zip(vehicles_df["vehicle_id"], vehicles_df["cost_basis"]))
    v_arrived= dict(zip(vehicles_df["vehicle_id"], vehicles_df["arrived_date"]))
    v_true   = dict(zip(vehicles_df["vehicle_id"], vehicles_df["_true_base_value"]))

    c_ids    = customers_df["customer_id"].values
    c_income = dict(zip(customers_df["customer_id"], customers_df["income_bracket"]))
    c_ps     = dict(zip(customers_df["customer_id"], customers_df["_price_sensitivity"]))

    r_ids       = reps_df["rep_id"].values
    r_archetype = dict(zip(reps_df["rep_id"], reps_df["_archetype"]))
    r_margin    = dict(zip(reps_df["rep_id"], reps_df["_margin_tendency"]))
    r_sat_bias  = {rid: _ARCHETYPE_PARAMS[RepArchetype(r_archetype[rid])]["sat_bias"] for rid in r_ids}

    rows = []
    for _ in range(n):
        vehicle_id  = str(rng.choice(v_ids))
        customer_id = str(rng.choice(c_ids))
        rep_id      = str(rng.choice(r_ids))
        sale_date   = _random_date(rng, date_start, date_end)

        asking    = float(v_asking[vehicle_id])
        cost      = float(v_cost[vehicle_id])
        true_base = float(v_true[vehicle_id])
        arrived   = _parse_date(str(v_arrived[vehicle_id]))

        price_premium = asking / max(true_base, 1.0)
        base_days = max(5, int(rng.normal(30, 12)))
        days_on_lot = int(np.clip(base_days * (price_premium ** 2.5), 3, 180))
        sale_date_adj = max(sale_date, arrived + timedelta(days=days_on_lot))
        if sale_date_adj > date_end:
            sale_date_adj = date_end

        ps = float(c_ps[customer_id])
        month_mult = _MONTH_DISCOUNT_MULTIPLIERS[sale_date.month]
        base_discount_pct = float(np.clip(rng.normal(0.04, 0.015) * ps * month_mult, 0.0, 0.18))
        discount_amount = round(asking * base_discount_pct, 2)
        sale_price = max(cost * 1.01, round(asking - discount_amount, -1))
        discount_amount = round(asking - sale_price, 2)
        discount_pct = round(discount_amount / asking, 4) if asking > 0 else 0.0

        archetype = r_archetype[rep_id]
        noise_scale = 800.0 if archetype == RepArchetype.STRUGGLING.value else 300.0
        gross_profit = round((sale_price - cost) + float(r_margin[rep_id]) + float(rng.normal(0, noise_scale)), 2)

        has_trade = rng.random() < 0.38
        trade_in_value = int(rng.integers(1_000, 22_000)) if has_trade else 0

        income_bracket = str(c_income[customer_id])
        fin_weights = _FINANCING_WEIGHTS_BY_INCOME.get(income_bracket, [0.2]*5)
        financing_type = str(rng.choice(_FINANCING_TYPES, p=fin_weights))

        raw_sat = rng.normal(3.6 + float(r_sat_bias[rep_id]), 0.7)
        sat_score = float(np.clip(round(raw_sat * 2) / 2, 1.0, 5.0))

        rows.append({
            "sale_id": _new_id(), "vehicle_id": vehicle_id,
            "customer_id": customer_id, "rep_id": rep_id,
            "sale_date": sale_date_adj.isoformat(),
            "asking_price_at_sale": int(asking), "sale_price": int(sale_price),
            "discount_amount": int(discount_amount), "discount_pct": discount_pct,
            "days_on_lot": days_on_lot, "trade_in_value": trade_in_value,
            "financing_type": financing_type, "gross_profit": int(gross_profit),
            "customer_satisfaction_score": sat_score,
            "_rep_archetype": archetype,
        })

    df = pd.DataFrame(rows)
    logger.success(
        f"Sales done — avg discount {df[\'discount_pct\'].mean():.1%}, "
        f"avg gross ${df[\'gross_profit\'].mean():,.0f}, "
        f"avg days-on-lot {df[\'days_on_lot\'].mean():.1f}"
    )
    return df
'''

# ======================================================================
# src/dealership/ingestion/synthetic/pipeline.py
# ======================================================================
files["src/dealership/ingestion/synthetic/pipeline.py"] = '''"""
Pipeline orchestrator — generate → validate → write Parquet.

Public files (data/raw/synthetic/):   no latent columns
Debug files  (data/debug/synthetic/): includes _* latent columns
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from dealership.common.config import get_settings
from dealership.ingestion.synthetic.generators import (
    generate_customers, generate_reps, generate_sales, generate_vehicles,
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
    rep_ids  = set(reps["rep_id"])
    cust_ids = set(customers["customer_id"])
    veh_ids  = set(vehicles["vehicle_id"])
    if bad := set(sales["rep_id"]) - rep_ids:
        errors.append(f"Sales reference {len(bad)} unknown rep_ids")
    if bad := set(sales["customer_id"]) - cust_ids:
        errors.append(f"Sales reference {len(bad)} unknown customer_ids")
    if bad := set(sales["vehicle_id"]) - veh_ids:
        errors.append(f"Sales reference {len(bad)} unknown vehicle_ids")
    merged = sales.merge(vehicles[["vehicle_id","cost_basis"]], on="vehicle_id", how="left")
    underwater = (merged["sale_price"] < merged["cost_basis"]).mean()
    if underwater > 0.10:
        errors.append(f"{underwater:.1%} of sales below cost basis (threshold 10%)")
    return errors


def run_pipeline(*, seed: int | None = None, num_reps: int | None = None,
                 num_customers: int | None = None, num_vehicles: int | None = None,
                 num_sales: int | None = None, raw_path: Path | None = None,
                 debug_path: Path | None = None, skip_debug: bool = False) -> dict[str, Any]:
    """Execute the full synthetic data pipeline."""
    seed    = seed if seed is not None else settings.synthetic_seed
    raw_out = raw_path or settings.raw_path()
    dbg_out = debug_path or settings.debug_path()

    logger.info("=" * 55)
    logger.info("Dealership Synthetic Data Pipeline — START")
    logger.info(f"  seed={seed}")
    logger.info("=" * 55)

    master = np.random.default_rng(seed)
    child  = lambda: np.random.default_rng(int(master.integers(0, 2**31)))

    reps      = generate_reps(n=num_reps,      rng=child())
    customers = generate_customers(n=num_customers, rng=child())
    vehicles  = generate_vehicles(n=num_vehicles,  rng=child())
    sales     = generate_sales(vehicles, customers, reps, n=num_sales, rng=child())

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
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "seed": seed,
        "record_counts": {k: len(v) for k, v in tables.items()},
        "date_range": {"start": settings.synthetic_date_start, "end": settings.synthetic_date_end},
        "archetype_mix": reps["_archetype"].value_counts().to_dict(),
        "condition_mix": vehicles["condition"].value_counts().to_dict(),
        "avg_sale_price": round(float(sales["sale_price"].mean()), 2),
        "avg_gross_profit": round(float(sales["gross_profit"].mean()), 2),
        "avg_days_on_lot": round(float(sales["days_on_lot"].mean()), 1),
        "december_avg_discount_pct": round(float(sales[sales_dt.dt.month==12]["discount_pct"].mean()), 4),
        "march_avg_discount_pct":    round(float(sales[sales_dt.dt.month==3]["discount_pct"].mean()), 4),
    }
    meta_path = raw_out / "run_metadata.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    logger.info(f"  Metadata -> {meta_path}")
    logger.success("Pipeline complete")
    return metadata
'''

# ======================================================================
# scripts/generate_synthetic_data.py
# ======================================================================
files["scripts/generate_synthetic_data.py"] = '''#!/usr/bin/env python
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
    logger.add(log_file,   format=fmt, level=level, rotation="10 MB", retention="7 days")


@app.command()
def main(
    seed:          int  = typer.Option(None,  "--seed"),
    num_reps:      int  = typer.Option(None,  "--num-reps"),
    num_customers: int  = typer.Option(None,  "--num-customers"),
    num_vehicles:  int  = typer.Option(None,  "--num-vehicles"),
    num_sales:     int  = typer.Option(None,  "--num-sales"),
    skip_debug:    bool = typer.Option(False, "--skip-debug"),
    log_level:     str  = typer.Option(None,  "--log-level"),
) -> None:
    """Run the synthetic data generation pipeline."""
    s = get_settings()
    _configure_logging((log_level or s.log_level).upper(), s.log_file)

    try:
        meta = run_pipeline(seed=seed, num_reps=num_reps, num_customers=num_customers,
                            num_vehicles=num_vehicles, num_sales=num_sales, skip_debug=skip_debug)
    except RuntimeError as exc:
        logger.error(f"Pipeline failed: {exc}")
        raise typer.Exit(code=1)

    typer.echo("")
    typer.secho("  Synthetic data generation complete", fg=typer.colors.GREEN, bold=True)
    typer.echo(f"  Records   : {meta[\'record_counts\']}")
    typer.echo(f"  Avg sale  : ${meta[\'avg_sale_price\']:,.0f}")
    typer.echo(f"  Avg gross : ${meta[\'avg_gross_profit\']:,.0f}")
    typer.echo(f"  Dec disc  : {meta[\'december_avg_discount_pct\']:.2%}")
    typer.echo(f"  Mar disc  : {meta[\'march_avg_discount_pct\']:.2%}")
    typer.echo(f"  Files     : {s.raw_path()}")


if __name__ == "__main__":
    app()
'''

# ======================================================================
# scripts/sanity_check_synthetic_data.py
# ======================================================================
files["scripts/sanity_check_synthetic_data.py"] = '''#!/usr/bin/env python
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
    raw_path:              Path  = typer.Option(None,  "--raw-path"),
    price_aging_threshold: float = typer.Option(0.10,  "--r-threshold"),
    dec_delta:             float = typer.Option(0.010, "--dec-delta"),
    rep_iqr:               float = typer.Option(500.0, "--rep-iqr"),
) -> None:
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
               colorize=True, level="DEBUG")

    s = get_settings()
    raw_dir = raw_path or s.raw_path()
    logger.info(f"Checking data at: {raw_dir}")

    try:
        reps      = pd.read_parquet(raw_dir / "reps.parquet")
        customers = pd.read_parquet(raw_dir / "customers.parquet")
        vehicles  = pd.read_parquet(raw_dir / "vehicles.parquet")
        sales     = pd.read_parquet(raw_dir / "sales.parquet")
    except FileNotFoundError as e:
        logger.error(f"Missing file: {e}. Run generate_synthetic_data.py first.")
        raise typer.Exit(1)

    logger.info(f"reps:{len(reps):,}  customers:{len(customers):,}  vehicles:{len(vehicles):,}  sales:{len(sales):,}")

    results = []

    # 1. No latent cols in public files
    for name, df in [("reps",reps),("customers",customers),("vehicles",vehicles),("sales",sales)]:
        latent = [c for c in df.columns if c.startswith("_")]
        results.append(_check(f"No latent cols in {name}.parquet", len(latent)==0,
                               "clean" if not latent else f"found: {latent}"))

    # 2. Referential integrity
    bad_r = set(sales["rep_id"])      - set(reps["rep_id"])
    bad_c = set(sales["customer_id"]) - set(customers["customer_id"])
    bad_v = set(sales["vehicle_id"])  - set(vehicles["vehicle_id"])
    all_ok = not (bad_r or bad_c or bad_v)
    results.append(_check("Referential integrity", all_ok,
        "all FKs resolve" if all_ok else f"orphans rep={len(bad_r)} cust={len(bad_c)} veh={len(bad_v)}"))

    # 3. Price <-> aging correlation
    r = sales["asking_price_at_sale"].corr(sales["days_on_lot"])
    results.append(_check("Price-aging correlation", r > price_aging_threshold,
                           f"Pearson r={r:.4f} (threshold>{price_aging_threshold})"))

    # 4. December discount > March
    dt = pd.to_datetime(sales["sale_date"])
    dec_d = sales[dt.dt.month==12]["discount_pct"].mean()
    mar_d = sales[dt.dt.month==3]["discount_pct"].mean()
    delta = dec_d - mar_d
    results.append(_check("December > March discount", delta > dec_delta,
                           f"Dec={dec_d:.4f} Mar={mar_d:.4f} delta={delta:.4f} (threshold>{dec_delta})"))

    # 5. Rep margin IQR
    per_rep = sales.groupby("rep_id")["gross_profit"].median()
    q25, q75 = per_rep.quantile(0.25), per_rep.quantile(0.75)
    iqr = q75 - q25
    results.append(_check("Rep margin spread IQR", iqr > rep_iqr,
                           f"IQR=${iqr:,.0f} Q25=${q25:,.0f} Q75=${q75:,.0f} (threshold>${rep_iqr:,.0f})"))

    logger.info("-" * 50)
    passed = sum(results)
    if passed == len(results):
        logger.success(f"All {passed}/{len(results)} checks passed!")
        raise typer.Exit(0)
    else:
        logger.error(f"{len(results)-passed}/{len(results)} checks FAILED")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
'''

# ======================================================================
# Write all files
# ======================================================================
created = []
for rel_path, content in files.items():
    target = ROOT / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    created.append(str(target))
    print(f"  OK  {rel_path}")

print(f"\nDone — {len(created)} files written.")
print("\nNext steps:")
print("  1. uv add typer loguru faker pyarrow dbt-duckdb")
print("  2. uv run python scripts/generate_synthetic_data.py")
print("  3. uv run python scripts/sanity_check_synthetic_data.py")
