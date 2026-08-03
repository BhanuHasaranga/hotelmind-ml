"""`DATA_SOURCE=postgres|local` abstraction over warehouse mart access.

Callers (RAG warehouse loader, insights rules) call `get_mart(name)` and
never need to know whether data comes from the live Postgres warehouse or
local parquet snapshots.
"""

import pandas as pd

from genai.config.genai_settings import genai_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_mart(mart_name: str) -> pd.DataFrame:
    source = genai_settings.DATA_SOURCE.lower()
    if source == "postgres":
        from genai.data_access.postgres_reader import read_mart as read_mart_postgres

        return read_mart_postgres(mart_name)
    if source == "local":
        from genai.data_access.local_reader import read_mart as read_mart_local

        return read_mart_local(mart_name)
    raise ValueError(f"Unknown DATA_SOURCE={source!r}; expected 'postgres' or 'local'")


def list_marts() -> list[str]:
    return [
        "mart_revenue_daily",
        "mart_occupancy_daily",
        "mart_restaurant_daily",
        "mart_staff_daily",
    ]
