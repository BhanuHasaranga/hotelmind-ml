from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values

import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class WarehouseWriteError(Exception):
    """Raised when a warehouse write cannot proceed (e.g. target table missing)."""


def table_exists(conn: PGConnection, table_name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
            "WHERE table_name = %s)",
            (table_name,),
        )
        return bool(cur.fetchone()[0])


def bulk_upsert(conn: PGConnection, table_name: str, df: pd.DataFrame, pk_cols: list[str]) -> int:
    """INSERT ... ON CONFLICT (pk_cols) DO UPDATE for every row in df.

    Raises WarehouseWriteError if table_name does not exist, so callers can
    log a clear message instead of a raw psycopg2 traceback.
    """
    if not table_exists(conn, table_name):
        raise WarehouseWriteError(
            f"Table '{table_name}' does not exist in the target database — "
            "skipping load. Verify the Phase 3 warehouse schema is provisioned."
        )

    if df.empty:
        return 0

    columns = list(df.columns)
    update_cols = [c for c in columns if c not in pk_cols]
    set_clause = ", ".join(f'"{c}" = EXCLUDED."{c}"' for c in update_cols)
    col_list = ", ".join(f'"{c}"' for c in columns)
    pk_list = ", ".join(f'"{c}"' for c in pk_cols)

    sql = f'INSERT INTO "{table_name}" ({col_list}) VALUES %s '
    sql += f"ON CONFLICT ({pk_list}) "
    sql += f"DO UPDATE SET {set_clause}" if update_cols else "DO NOTHING"

    values = [tuple(row) for row in df.itertuples(index=False, name=None)]

    with conn.cursor() as cur:
        execute_values(cur, sql, values)

    logger.info("Upserted %d rows into %s", len(values), table_name)
    return len(values)
