"""Local SQLite persistence, TTL caching, and privacy-conscious record management."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from alt_detector.exceptions import StorageError
from alt_detector.schemas import AccountObservation, LabeledAccountPair


class SQLiteStorage:
    """Thread-safe local SQLite storage for caching, derived observations, and labels.

    Designed for privacy-conscious local operation:
    - Stores derived observations and bounded API responses with explicit TTL expiration.
    - Provides deterministic deletion of specific user records or the entire cache.
    - Never stores credentials, cookies, or session tokens.
    """

    def __init__(
        self,
        db_path: Path | str,
        *,
        default_ttl_seconds: int = 3600,
        max_cache_entries: int = 5000,
    ) -> None:
        self.db_path = Path(db_path)
        self.default_ttl_seconds = max(0, int(default_ttl_seconds))
        self.max_cache_entries = max(10, int(max_cache_entries))
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = None
        self._initialize_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            try:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                conn = sqlite3.connect(
                    str(self.db_path),
                    timeout=10.0,
                    check_same_thread=False,
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA foreign_keys=ON;")
                self._conn = conn
            except sqlite3.Error as exc:
                raise StorageError(f"Failed to open SQLite database at {self.db_path}: {exc}") from exc
        return self._conn

    def _initialize_db(self) -> None:
        with self._lock:
            conn = self._get_conn()
            try:
                with conn:
                    conn.executescript(
                        """
                        CREATE TABLE IF NOT EXISTS http_cache (
                            cache_key TEXT PRIMARY KEY,
                            source_name TEXT NOT NULL,
                            endpoint_url TEXT NOT NULL,
                            http_status INTEGER NOT NULL,
                            payload_json TEXT NOT NULL,
                            retrieved_at TEXT NOT NULL,
                            created_epoch REAL NOT NULL,
                            expires_epoch REAL NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_http_cache_expires
                            ON http_cache(expires_epoch);

                        CREATE TABLE IF NOT EXISTS account_observations (
                            user_id INTEGER PRIMARY KEY,
                            username TEXT NOT NULL,
                            display_name TEXT NOT NULL,
                            observation_json TEXT NOT NULL,
                            retrieved_at TEXT NOT NULL,
                            expires_epoch REAL NOT NULL
                        );
                        CREATE INDEX IF NOT EXISTS idx_account_obs_username
                            ON account_observations(LOWER(username));

                        CREATE TABLE IF NOT EXISTS labeled_pairs (
                            min_user_id INTEGER NOT NULL,
                            max_user_id INTEGER NOT NULL,
                            pair_json TEXT NOT NULL,
                            label INTEGER NOT NULL,
                            operator_cluster_id TEXT NOT NULL,
                            adjudication_status TEXT NOT NULL,
                            updated_at TEXT NOT NULL,
                            PRIMARY KEY (min_user_id, max_user_id)
                        );
                        """
                    )
            except sqlite3.Error as exc:
                raise StorageError(f"Failed to initialize SQLite schema: {exc}") from exc

    @staticmethod
    def make_cache_key(
        method: str,
        url: str,
        json_body: dict[str, Any] | None = None,
    ) -> str:
        """Compute a canonical SHA-256 key for an HTTP request."""
        body_part = (
            json.dumps(json_body, sort_keys=True, separators=(",", ":"))
            if json_body is not None
            else ""
        )
        raw = f"{method.upper()}|{url}|{body_part}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get_cached_response(
        self,
        cache_key: str,
        *,
        now_epoch: float | None = None,
    ) -> dict[str, Any] | None:
        """Retrieve a cached HTTP response if present and not expired."""
        now = time.time() if now_epoch is None else now_epoch
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                """
                SELECT source_name, endpoint_url, http_status, payload_json, retrieved_at, expires_epoch
                FROM http_cache
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
            if row is None:
                return None
            if float(row["expires_epoch"]) <= now:
                with conn:
                    conn.execute("DELETE FROM http_cache WHERE cache_key = ?", (cache_key,))
                return None
            try:
                payload = json.loads(row["payload_json"])
            except json.JSONDecodeError:
                return None
            return {
                "source_name": row["source_name"],
                "endpoint_url": row["endpoint_url"],
                "http_status": int(row["http_status"]),
                "payload": payload,
                "retrieved_at": row["retrieved_at"],
                "from_cache": True,
            }

    def set_cached_response(
        self,
        cache_key: str,
        *,
        source_name: str,
        endpoint_url: str,
        http_status: int,
        payload: Any,
        ttl_seconds: int | None = None,
        now_epoch: float | None = None,
    ) -> str:
        """Store a minimized JSON response in the local SQLite cache with TTL."""
        ttl = self.default_ttl_seconds if ttl_seconds is None else max(0, int(ttl_seconds))
        now = time.time() if now_epoch is None else now_epoch
        retrieved_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        if ttl == 0:
            return retrieved_iso

        expires = now + ttl
        payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO http_cache
                    (cache_key, source_name, endpoint_url, http_status, payload_json, retrieved_at, created_epoch, expires_epoch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cache_key,
                        source_name,
                        endpoint_url,
                        int(http_status),
                        payload_str,
                        retrieved_iso,
                        now,
                        expires,
                    ),
                )
            self._enforce_max_entries_locked(conn)
        return retrieved_iso

    def _enforce_max_entries_locked(self, conn: sqlite3.Connection) -> None:
        count_row = conn.execute("SELECT COUNT(*) AS cnt FROM http_cache").fetchone()
        total = int(count_row["cnt"]) if count_row else 0
        if total > self.max_cache_entries:
            excess = total - self.max_cache_entries
            with conn:
                conn.execute(
                    """
                    DELETE FROM http_cache
                    WHERE cache_key IN (
                        SELECT cache_key FROM http_cache
                        ORDER BY created_epoch ASC
                        LIMIT ?
                    )
                    """,
                    (excess,),
                )

    def purge_expired(self, *, now_epoch: float | None = None) -> int:
        """Remove all expired cache and account observation rows."""
        now = time.time() if now_epoch is None else now_epoch
        with self._lock:
            conn = self._get_conn()
            with conn:
                cur1 = conn.execute("DELETE FROM http_cache WHERE expires_epoch <= ?", (now,))
                cur2 = conn.execute(
                    "DELETE FROM account_observations WHERE expires_epoch <= ?",
                    (now,),
                )
                return (cur1.rowcount or 0) + (cur2.rowcount or 0)

    def save_account_observation(
        self,
        observation: AccountObservation,
        *,
        ttl_seconds: int | None = None,
        now_epoch: float | None = None,
    ) -> None:
        """Persist a derived AccountObservation summary with TTL."""
        ttl = self.default_ttl_seconds if ttl_seconds is None else max(0, int(ttl_seconds))
        if ttl == 0:
            return
        now = time.time() if now_epoch is None else now_epoch
        retrieved_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
        expires = now + ttl
        obs_json = observation.model_dump_json()
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO account_observations
                    (user_id, username, display_name, observation_json, retrieved_at, expires_epoch)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        observation.user_id,
                        observation.username,
                        observation.display_name,
                        obs_json,
                        retrieved_iso,
                        expires,
                    ),
                )

    def get_account_observation(
        self,
        user_id: int,
        *,
        now_epoch: float | None = None,
    ) -> AccountObservation | None:
        """Retrieve a cached derived AccountObservation if fresh."""
        now = time.time() if now_epoch is None else now_epoch
        with self._lock:
            conn = self._get_conn()
            row = conn.execute(
                """
                SELECT observation_json, expires_epoch
                FROM account_observations
                WHERE user_id = ?
                """,
                (int(user_id),),
            ).fetchone()
            if row is None:
                return None
            if float(row["expires_epoch"]) <= now:
                with conn:
                    conn.execute(
                        "DELETE FROM account_observations WHERE user_id = ?",
                        (int(user_id),),
                    )
                return None
            return AccountObservation.model_validate_json(row["observation_json"])

    def delete_account_records(self, identifier: str | int) -> int:
        """Delete locally stored records for a specific user ID or username (Privacy control)."""
        ident_str = str(identifier).strip()
        deleted = 0
        with self._lock:
            conn = self._get_conn()
            with conn:
                if ident_str.isdigit():
                    uid = int(ident_str)
                    row = conn.execute(
                        "SELECT username FROM account_observations WHERE user_id = ?",
                        (uid,),
                    ).fetchone()
                    uname = row["username"] if row else None
                    c1 = conn.execute(
                        "DELETE FROM account_observations WHERE user_id = ?",
                        (uid,),
                    )
                    deleted += c1.rowcount or 0
                    c2 = conn.execute(
                        "DELETE FROM http_cache WHERE endpoint_url LIKE ? OR payload_json LIKE ?",
                        (f"%/{uid}%", f"%{uid}%"),
                    )
                    deleted += c2.rowcount or 0
                    if uname:
                        c3 = conn.execute(
                            "DELETE FROM http_cache WHERE payload_json LIKE ?",
                            (f"%{uname}%",),
                        )
                        deleted += c3.rowcount or 0
                else:
                    row = conn.execute(
                        "SELECT user_id FROM account_observations WHERE LOWER(username) = LOWER(?)",
                        (ident_str,),
                    ).fetchone()
                    uid = int(row["user_id"]) if row else None
                    c1 = conn.execute(
                        "DELETE FROM account_observations WHERE LOWER(username) = LOWER(?)",
                        (ident_str,),
                    )
                    deleted += c1.rowcount or 0
                    c2 = conn.execute(
                        "DELETE FROM http_cache WHERE LOWER(payload_json) LIKE LOWER(?)",
                        (f"%{ident_str}%",),
                    )
                    deleted += c2.rowcount or 0
                    if uid is not None:
                        c3 = conn.execute(
                            "DELETE FROM http_cache WHERE endpoint_url LIKE ?",
                            (f"%/{uid}%",),
                        )
                        deleted += c3.rowcount or 0
        return deleted

    def clear_all_cache(self) -> int:
        """Delete all cached HTTP responses and derived account observations."""
        with self._lock:
            conn = self._get_conn()
            with conn:
                c1 = conn.execute("DELETE FROM http_cache")
                c2 = conn.execute("DELETE FROM account_observations")
                return (c1.rowcount or 0) + (c2.rowcount or 0)

    def upsert_labeled_pair(self, pair: LabeledAccountPair) -> None:
        """Insert or update a ground-truth / weakly-labeled account pair with deduplication."""
        min_uid = min(pair.user_id_a, pair.user_id_b)
        max_uid = max(pair.user_id_a, pair.user_id_b)
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        with self._lock:
            conn = self._get_conn()
            with conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO labeled_pairs
                    (min_user_id, max_user_id, pair_json, label, operator_cluster_id, adjudication_status, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        min_uid,
                        max_uid,
                        pair.model_dump_json(),
                        pair.label,
                        pair.operator_cluster_id,
                        pair.adjudication_status,
                        now_iso,
                    ),
                )

    def load_labeled_pairs(self) -> list[LabeledAccountPair]:
        """Load all stored labeled pairs in deterministic order."""
        with self._lock:
            conn = self._get_conn()
            rows = conn.execute(
                "SELECT pair_json FROM labeled_pairs ORDER BY min_user_id ASC, max_user_id ASC"
            ).fetchall()
            return [LabeledAccountPair.model_validate_json(r["pair_json"]) for r in rows]

    def stats(self) -> dict[str, int]:
        """Return current row counts for local storage tables."""
        with self._lock:
            conn = self._get_conn()
            http_cnt = conn.execute("SELECT COUNT(*) FROM http_cache").fetchone()[0]
            obs_cnt = conn.execute("SELECT COUNT(*) FROM account_observations").fetchone()[0]
            lbl_cnt = conn.execute("SELECT COUNT(*) FROM labeled_pairs").fetchone()[0]
            return {
                "http_cache_entries": int(http_cnt),
                "account_observations": int(obs_cnt),
                "labeled_pairs": int(lbl_cnt),
            }

    def close(self) -> None:
        """Close the underlying SQLite connection cleanly."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> SQLiteStorage:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

