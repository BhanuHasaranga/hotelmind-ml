from pathlib import Path

import pandas as pd

from src.database.postgres import get_connection
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run_query(sql_path: Path, params: dict | None = None) -> pd.DataFrame:
    """Execute a .sql file from sql/ against the warehouse and return a DataFrame.

    This is the only place SQL execution glue lives; domain modules pass a path
    under sql/ plus bind params, never raw SQL strings.
    """
    sql_text = sql_path.read_text(encoding="utf-8")
    logger.info("Running query %s with params=%s", sql_path.name, params)
    conn = get_connection()
    try:
        return pd.read_sql(sql_text, conn, params=params)
    finally:
        conn.close()
