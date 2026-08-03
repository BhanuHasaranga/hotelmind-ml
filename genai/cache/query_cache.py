"""Disk-backed cache (SQLite, stdlib only) shared by embeddings and RAG
query results.

Two logical namespaces share one table, keyed by a caller-supplied
`namespace` (e.g. "embedding", "rag_query"): a content-hash-keyed cache for
embeddings (no TTL needed — content hash already invalidates on change) and
a TTL-bound cache for repeated RAG query results.
"""

import json
import sqlite3
import time
from pathlib import Path

from genai.config.genai_settings import genai_settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    namespace TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL,
    PRIMARY KEY (namespace, key)
)
"""


class QueryCache:
    """Simple SQLite key/value cache with optional per-entry TTL."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path or genai_settings.cache_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def get(self, namespace: str, key: str) -> object | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value, expires_at FROM cache_entries WHERE namespace = ? AND key = ?",
                (namespace, key),
            ).fetchone()
        if row is None:
            return None
        value, expires_at = row
        if expires_at is not None and expires_at < time.time():
            self.delete(namespace, key)
            return None
        return json.loads(value)

    def set(self, namespace: str, key: str, value: object, ttl_seconds: int | None = None) -> None:
        expires_at = time.time() + ttl_seconds if ttl_seconds else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cache_entries (namespace, key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                    value = excluded.value,
                    created_at = excluded.created_at,
                    expires_at = excluded.expires_at
                """,
                (namespace, key, json.dumps(value), time.time(), expires_at),
            )
            conn.commit()

    def delete(self, namespace: str, key: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM cache_entries WHERE namespace = ? AND key = ?", (namespace, key)
            )
            conn.commit()

    def clear_namespace(self, namespace: str) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM cache_entries WHERE namespace = ?", (namespace,))
            conn.commit()
            return cur.rowcount

    def stats(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT namespace, COUNT(*) FROM cache_entries GROUP BY namespace"
            ).fetchall()
        return {namespace: count for namespace, count in rows}


_default_cache: QueryCache | None = None


def get_cache() -> QueryCache:
    global _default_cache
    if _default_cache is None:
        _default_cache = QueryCache()
    return _default_cache
