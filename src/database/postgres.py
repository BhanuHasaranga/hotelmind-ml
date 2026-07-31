import psycopg2
from psycopg2.extensions import connection as PGConnection

from src.config.settings import settings


def get_connection() -> PGConnection:
    return psycopg2.connect(
        host=settings.WAREHOUSE_DB_HOST,
        port=settings.WAREHOUSE_DB_PORT,
        dbname=settings.WAREHOUSE_DB_NAME,
        user=settings.WAREHOUSE_DB_USER,
        password=settings.WAREHOUSE_DB_PASSWORD,
    )
