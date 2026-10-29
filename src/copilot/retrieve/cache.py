"""SQLite cache for embeddings and rerank scores, so nothing is paid for twice.

Each entry also stores what the original API call cost (latency, characters, tokens), so a cached
rerun can still report the latency and usage a live system would see.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import numpy as np

CACHE_PATH = Path("data/cache/retrieval.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key TEXT PRIMARY KEY, vec BLOB NOT NULL, api_latency_s REAL NOT NULL, chars INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rerank (
    key TEXT PRIMARY KEY, payload TEXT NOT NULL
);
"""


def digest(*parts: object) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, path: Path = CACHE_PATH) -> None:
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        # The API serves requests from a thread pool, and the daily evaluation job may write to the
        # same file from another process: share one connection behind a lock and wait on locks.
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self._lock = threading.Lock()
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # embeddings
    def get_embedding(self, key: str) -> tuple[np.ndarray, float, int] | None:
        with self._lock:
            row = self._db.execute(
                "SELECT vec, api_latency_s, chars FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return np.frombuffer(row[0], dtype=np.float32), row[1], row[2]

    def put_embeddings(self, rows: list[tuple[str, np.ndarray, float, int]]) -> None:
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)",
                [(k, v.astype(np.float32).tobytes(), lat, chars) for k, v, lat, chars in rows],
            )
            self._db.commit()

    # rerank
    def get_rerank(self, key: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT payload FROM rerank WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put_rerank(self, key: str, payload: dict) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO rerank VALUES (?, ?)", (key, json.dumps(payload))
            )
            self._db.commit()
