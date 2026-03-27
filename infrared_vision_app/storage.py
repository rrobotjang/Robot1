from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Iterable

from models import Camera, Event, Site


def _dump_model(model) -> dict:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    return model.dict()


class InfraredVisionStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        default_path = Path(__file__).resolve().parent / "data" / "infrared_vision.db"
        self.path = Path(db_path or os.getenv("APP_DB_PATH") or default_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS sites (
                    id TEXT PRIMARY KEY,
                    profile TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cameras (
                    id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    site_id TEXT NOT NULL,
                    camera_id TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS audit_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    action TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    user_role TEXT NOT NULL,
                    refresh_token_hash TEXT NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_used_at TEXT NOT NULL,
                    revoked_at TEXT,
                    user_agent TEXT,
                    ip_address TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_events_site_occurred_at
                    ON events (site_id, occurred_at DESC);

                CREATE INDEX IF NOT EXISTS idx_events_status
                    ON events (status);

                CREATE INDEX IF NOT EXISTS idx_sessions_username
                    ON user_sessions (username, revoked_at, expires_at);
                """
            )

    def seed_if_empty(
        self,
        *,
        sites: Iterable[Site],
        cameras: Iterable[Camera],
        events: Iterable[Event],
    ) -> None:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) AS count FROM sites").fetchone()["count"]
            if count:
                return

        for site in sites:
            self.upsert_site(site)
        for camera in cameras:
            self.upsert_camera(camera)
        for event in events:
            self.upsert_event(event)

    def upsert_site(self, site: Site) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sites (id, profile, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    profile = excluded.profile,
                    payload = excluded.payload
                """,
                (site.id, site.profile.value, json.dumps(_dump_model(site), ensure_ascii=False)),
            )

    def upsert_camera(self, camera: Camera) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cameras (id, site_id, payload)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    site_id = excluded.site_id,
                    payload = excluded.payload
                """,
                (camera.id, camera.site_id, json.dumps(_dump_model(camera), ensure_ascii=False)),
            )

    def upsert_event(self, event: Event) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO events (id, site_id, camera_id, occurred_at, status, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    site_id = excluded.site_id,
                    camera_id = excluded.camera_id,
                    occurred_at = excluded.occurred_at,
                    status = excluded.status,
                    payload = excluded.payload
                """,
                (
                    event.id,
                    event.site_id,
                    event.camera_id,
                    event.occurred_at.isoformat(),
                    event.status.value,
                    json.dumps(_dump_model(event), ensure_ascii=False),
                ),
            )

    def fetch_sites(self, profile: str | None = None) -> list[dict]:
        query = "SELECT payload FROM sites"
        params: tuple = ()
        if profile:
            query += " WHERE profile = ?"
            params = (profile,)
        query += " ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def fetch_site(self, site_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM sites WHERE id = ?", (site_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def fetch_cameras(self, site_ids: list[str]) -> list[dict]:
        if not site_ids:
            return []
        placeholders = ",".join("?" for _ in site_ids)
        query = f"SELECT payload FROM cameras WHERE site_id IN ({placeholders}) ORDER BY id"
        with self._connect() as conn:
            rows = conn.execute(query, tuple(site_ids)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def fetch_camera(self, camera_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM cameras WHERE id = ?", (camera_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def fetch_events(self, site_ids: list[str]) -> list[dict]:
        if not site_ids:
            return []
        placeholders = ",".join("?" for _ in site_ids)
        query = (
            f"SELECT payload FROM events WHERE site_id IN ({placeholders}) "
            "ORDER BY occurred_at DESC"
        )
        with self._connect() as conn:
            rows = conn.execute(query, tuple(site_ids)).fetchall()
        return [json.loads(row["payload"]) for row in rows]

    def fetch_event(self, event_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM events WHERE id = ?", (event_id,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def insert_audit_log(self, action: str, actor: str, payload: dict, created_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO audit_logs (action, actor, created_at, payload)
                VALUES (?, ?, ?, ?)
                """,
                (action, actor, created_at, json.dumps(payload, ensure_ascii=False)),
            )

    def fetch_audit_logs(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, action, actor, created_at, payload
                FROM audit_logs
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "action": row["action"],
                "actor": row["actor"],
                "created_at": row["created_at"],
                "payload": json.loads(row["payload"]),
            }
            for row in rows
        ]

    def count_events(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM events").fetchone()
        return int(row["count"])

    def count_audit_logs(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS count FROM audit_logs").fetchone()
        return int(row["count"])

    def upsert_session(
        self,
        *,
        session_id: str,
        username: str,
        user_role: str,
        refresh_token_hash: str,
        issued_at: str,
        expires_at: str,
        last_used_at: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
        revoked_at: str | None = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_sessions (
                    session_id, username, user_role, refresh_token_hash,
                    issued_at, expires_at, last_used_at, revoked_at, user_agent, ip_address
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    username = excluded.username,
                    user_role = excluded.user_role,
                    refresh_token_hash = excluded.refresh_token_hash,
                    issued_at = excluded.issued_at,
                    expires_at = excluded.expires_at,
                    last_used_at = excluded.last_used_at,
                    revoked_at = excluded.revoked_at,
                    user_agent = excluded.user_agent,
                    ip_address = excluded.ip_address
                """,
                (
                    session_id,
                    username,
                    user_role,
                    refresh_token_hash,
                    issued_at,
                    expires_at,
                    last_used_at,
                    revoked_at,
                    user_agent,
                    ip_address,
                ),
            )

    def fetch_session(self, session_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, username, user_role, refresh_token_hash,
                       issued_at, expires_at, last_used_at, revoked_at, user_agent, ip_address
                FROM user_sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def revoke_session(self, session_id: str, revoked_at: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE user_sessions
                SET revoked_at = ?, last_used_at = ?
                WHERE session_id = ?
                """,
                (revoked_at, revoked_at, session_id),
            )

    def count_active_sessions(self, now_iso: str) -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM user_sessions
                WHERE revoked_at IS NULL
                  AND expires_at >= ?
                """,
                (now_iso,),
            ).fetchone()
        return int(row["count"])
