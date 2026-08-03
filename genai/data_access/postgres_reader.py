"""Reads mart tables directly from the live `hotelmind_warehouse` Postgres
DB for `DATA_SOURCE=postgres`.

Reuses `src.config.settings.settings.WAREHOUSE_DB_URL` for connection
credentials rather than duplicating DB config in `genai_settings`.
"""

import pandas as pd

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_VALID_MARTS = {
    "mart_revenue_daily",
    "mart_occupancy_daily",
    "mart_restaurant_daily",
    "mart_staff_daily",
}


def read_mart(mart_name: str, limit: int | None = None) -> pd.DataFrame:
    if mart_name not in _VALID_MARTS:
        raise ValueError(f"Unknown mart: {mart_name}; expected one of {sorted(_VALID_MARTS)}")

    try:
        import psycopg2
    except ImportError as exc:  # pragma: no cover - psycopg2-binary is a hard dependency
        raise RuntimeError("psycopg2 is required for DATA_SOURCE=postgres") from exc

    query = f"SELECT * FROM {mart_name}"
    if limit:
        query += f" LIMIT {int(limit)}"

    try:
        conn = psycopg2.connect(settings.WAREHOUSE_DB_URL, connect_timeout=3)
    except psycopg2.OperationalError as exc:
        logger.error("Could not connect to warehouse DB: %s", exc)
        raise ConnectionError(f"Unable to connect to warehouse database: {exc}") from exc

    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()
