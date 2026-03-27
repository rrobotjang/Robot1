from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Protocol

from models import Camera, Event, Site
from storage import InfraredVisionStore

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # noqa: BLE001
    psycopg = None
    dict_row = None


def _dump_model(model) -> dict:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    return model.dict()


class VisionStoreProtocol(Protocol):
    path: str | Path

    def seed_if_empty(self, *, sites: Iterable[Site], cameras: Iterable[Camera], events: Iterable[Event]) -> None: ...
    def upsert_site(self, site: Site) -> None: ...
    def upsert_camera(self, camera: Camera) -> None: ...
    def upsert_event(self, event: Event) -> None: ...
    def fetch_sites(self, profile: str | None = None) -> list[dict]: ...
    def fetch_site(self, site_id: str) -> dict | None: ...
    def fetch_cameras(self, site_ids: list[str]) -> list[dict]: ...
    def fetch_camera(self, camera_id: str) -> dict | None: ...
    def fetch_events(self, site_ids: list[str]) -> list[dict]: ...
    def fetch_event(self, event_id: str) -> dict | None: ...
    def insert_audit_log(self, action: str, actor: str, payload: dict, created_at: str) -> None: ...
    def fetch_audit_logs(self, limit: int = 50) -> list[dict]: ...
    def count_events(self) -> int: ...
    def count_audit_logs(self) -> int: ...
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
    ) -> None: ...
    def fetch_session(self, session_id: str) -> dict | None: ...
    def revoke_session(self, session_id: str, revoked_at: str) -> None: ...
    def count_active_sessions(self, now_iso: str) -> int: ...


class MemoryVisionStore:
    def __init__(self) -> None:
        self.path = "memory://infrared_vision"
        self._sites: dict[str, dict] = {}
        self._cameras: dict[str, dict] = {}
        self._events: dict[str, dict] = {}
        self._audit_logs: list[dict] = []
        self._sessions: dict[str, dict] = {}
        self._audit_counter = 1

    def seed_if_empty(self, *, sites: Iterable[Site], cameras: Iterable[Camera], events: Iterable[Event]) -> None:
        if self._sites:
            return
        for site in sites:
            self.upsert_site(site)
        for camera in cameras:
            self.upsert_camera(camera)
        for event in events:
            self.upsert_event(event)

    def upsert_site(self, site: Site) -> None:
        self._sites[site.id] = _dump_model(site)

    def upsert_camera(self, camera: Camera) -> None:
        self._cameras[camera.id] = _dump_model(camera)

    def upsert_event(self, event: Event) -> None:
        self._events[event.id] = _dump_model(event)

    def fetch_sites(self, profile: str | None = None) -> list[dict]:
        rows = list(self._sites.values())
        if profile:
            rows = [row for row in rows if row["profile"] == profile]
        return sorted(rows, key=lambda item: item["id"])

    def fetch_site(self, site_id: str) -> dict | None:
        return self._sites.get(site_id)

    def fetch_cameras(self, site_ids: list[str]) -> list[dict]:
        return [row for row in self._cameras.values() if row["site_id"] in site_ids]

    def fetch_camera(self, camera_id: str) -> dict | None:
        return self._cameras.get(camera_id)

    def fetch_events(self, site_ids: list[str]) -> list[dict]:
        rows = [row for row in self._events.values() if row["site_id"] in site_ids]
        return sorted(rows, key=lambda item: item["occurred_at"], reverse=True)

    def fetch_event(self, event_id: str) -> dict | None:
        return self._events.get(event_id)

    def insert_audit_log(self, action: str, actor: str, payload: dict, created_at: str) -> None:
        self._audit_logs.append(
            {
                "id": self._audit_counter,
                "action": action,
                "actor": actor,
                "created_at": created_at,
                "payload": payload,
            }
        )
        self._audit_counter += 1

    def fetch_audit_logs(self, limit: int = 50) -> list[dict]:
        return list(reversed(self._audit_logs))[:limit]

    def count_events(self) -> int:
        return len(self._events)

    def count_audit_logs(self) -> int:
        return len(self._audit_logs)

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
        self._sessions[session_id] = {
            "session_id": session_id,
            "username": username,
            "user_role": user_role,
            "refresh_token_hash": refresh_token_hash,
            "issued_at": issued_at,
            "expires_at": expires_at,
            "last_used_at": last_used_at,
            "revoked_at": revoked_at,
            "user_agent": user_agent,
            "ip_address": ip_address,
        }

    def fetch_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def revoke_session(self, session_id: str, revoked_at: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        session["revoked_at"] = revoked_at
        session["last_used_at"] = revoked_at

    def count_active_sessions(self, now_iso: str) -> int:
        now = datetime.fromisoformat(now_iso)
        active = 0
        for session in self._sessions.values():
            if session.get("revoked_at"):
                continue
            expires_at = session["expires_at"]
            expires_dt = expires_at if isinstance(expires_at, datetime) else datetime.fromisoformat(expires_at)
            if expires_dt >= now:
                active += 1
        return active


class PostgreSQLVisionStore:
    def __init__(self, dsn: str) -> None:
        if psycopg is None:
            raise RuntimeError("psycopg is not installed")
        self.path = dsn
        self.dsn = dsn
        self._ensure_schema()

    def _connect(self):
        return psycopg.connect(self.dsn, autocommit=True, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE SCHEMA IF NOT EXISTS infrared_vision")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS infrared_vision.sites (
                        id TEXT PRIMARY KEY,
                        profile TEXT NOT NULL,
                        payload JSONB NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS infrared_vision.cameras (
                        id TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        payload JSONB NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS infrared_vision.events (
                        id TEXT PRIMARY KEY,
                        site_id TEXT NOT NULL,
                        camera_id TEXT NOT NULL,
                        occurred_at TIMESTAMPTZ NOT NULL,
                        status TEXT NOT NULL,
                        payload JSONB NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS infrared_vision.audit_logs (
                        id BIGSERIAL PRIMARY KEY,
                        action TEXT NOT NULL,
                        actor TEXT NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL,
                        payload JSONB NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS infrared_vision.user_sessions (
                        session_id TEXT PRIMARY KEY,
                        username TEXT NOT NULL,
                        user_role TEXT NOT NULL,
                        refresh_token_hash TEXT NOT NULL,
                        issued_at TIMESTAMPTZ NOT NULL,
                        expires_at TIMESTAMPTZ NOT NULL,
                        last_used_at TIMESTAMPTZ NOT NULL,
                        revoked_at TIMESTAMPTZ NULL,
                        user_agent TEXT NULL,
                        ip_address TEXT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_events_site_occurred_at
                    ON infrared_vision.events (site_id, occurred_at DESC)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_events_status
                    ON infrared_vision.events (status)
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_sessions_username
                    ON infrared_vision.user_sessions (username, revoked_at, expires_at)
                    """
                )

    def seed_if_empty(self, *, sites: Iterable[Site], cameras: Iterable[Camera], events: Iterable[Event]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM infrared_vision.sites")
                row = cur.fetchone()
                if row["count"]:
                    return
        for site in sites:
            self.upsert_site(site)
        for camera in cameras:
            self.upsert_camera(camera)
        for event in events:
            self.upsert_event(event)

    def upsert_site(self, site: Site) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO infrared_vision.sites (id, profile, payload)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        profile = EXCLUDED.profile,
                        payload = EXCLUDED.payload
                    """,
                    (site.id, site.profile.value, json.dumps(_dump_model(site), ensure_ascii=False)),
                )

    def upsert_camera(self, camera: Camera) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO infrared_vision.cameras (id, site_id, payload)
                    VALUES (%s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        site_id = EXCLUDED.site_id,
                        payload = EXCLUDED.payload
                    """,
                    (camera.id, camera.site_id, json.dumps(_dump_model(camera), ensure_ascii=False)),
                )

    def upsert_event(self, event: Event) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO infrared_vision.events (id, site_id, camera_id, occurred_at, status, payload)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (id) DO UPDATE SET
                        site_id = EXCLUDED.site_id,
                        camera_id = EXCLUDED.camera_id,
                        occurred_at = EXCLUDED.occurred_at,
                        status = EXCLUDED.status,
                        payload = EXCLUDED.payload
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
        query = "SELECT payload FROM infrared_vision.sites"
        params: list = []
        if profile:
            query += " WHERE profile = %s"
            params.append(profile)
        query += " ORDER BY id"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        return [row["payload"] for row in rows]

    def fetch_site(self, site_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM infrared_vision.sites WHERE id = %s", (site_id,))
                row = cur.fetchone()
        return row["payload"] if row else None

    def fetch_cameras(self, site_ids: list[str]) -> list[dict]:
        if not site_ids:
            return []
        query = "SELECT payload FROM infrared_vision.cameras WHERE site_id = ANY(%s) ORDER BY id"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (site_ids,))
                rows = cur.fetchall()
        return [row["payload"] for row in rows]

    def fetch_camera(self, camera_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM infrared_vision.cameras WHERE id = %s", (camera_id,))
                row = cur.fetchone()
        return row["payload"] if row else None

    def fetch_events(self, site_ids: list[str]) -> list[dict]:
        if not site_ids:
            return []
        query = "SELECT payload FROM infrared_vision.events WHERE site_id = ANY(%s) ORDER BY occurred_at DESC"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (site_ids,))
                rows = cur.fetchall()
        return [row["payload"] for row in rows]

    def fetch_event(self, event_id: str) -> dict | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT payload FROM infrared_vision.events WHERE id = %s", (event_id,))
                row = cur.fetchone()
        return row["payload"] if row else None

    def insert_audit_log(self, action: str, actor: str, payload: dict, created_at: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO infrared_vision.audit_logs (action, actor, created_at, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (action, actor, created_at, json.dumps(payload, ensure_ascii=False)),
                )

    def fetch_audit_logs(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, action, actor, created_at, payload
                    FROM infrared_vision.audit_logs
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return rows

    def count_events(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM infrared_vision.events")
                row = cur.fetchone()
        return int(row["count"])

    def count_audit_logs(self) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) AS count FROM infrared_vision.audit_logs")
                row = cur.fetchone()
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO infrared_vision.user_sessions (
                        session_id, username, user_role, refresh_token_hash,
                        issued_at, expires_at, last_used_at, revoked_at, user_agent, ip_address
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        username = EXCLUDED.username,
                        user_role = EXCLUDED.user_role,
                        refresh_token_hash = EXCLUDED.refresh_token_hash,
                        issued_at = EXCLUDED.issued_at,
                        expires_at = EXCLUDED.expires_at,
                        last_used_at = EXCLUDED.last_used_at,
                        revoked_at = EXCLUDED.revoked_at,
                        user_agent = EXCLUDED.user_agent,
                        ip_address = EXCLUDED.ip_address
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
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT session_id, username, user_role, refresh_token_hash,
                           issued_at, expires_at, last_used_at, revoked_at, user_agent, ip_address
                    FROM infrared_vision.user_sessions
                    WHERE session_id = %s
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
        return row if row else None

    def revoke_session(self, session_id: str, revoked_at: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE infrared_vision.user_sessions
                    SET revoked_at = %s, last_used_at = %s
                    WHERE session_id = %s
                    """,
                    (revoked_at, revoked_at, session_id),
                )

    def count_active_sessions(self, now_iso: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM infrared_vision.user_sessions
                    WHERE revoked_at IS NULL AND expires_at >= %s
                    """,
                    (now_iso,),
                )
                row = cur.fetchone()
        return int(row["count"])


def create_store(
    *,
    db_backend: str,
    db_path: str | Path | None = None,
    db_url: str | None = None,
) -> VisionStoreProtocol:
    if db_backend == "memory":
        return MemoryVisionStore()
    if db_backend == "postgresql":
        if not db_url:
            raise ValueError("DB_URL is required when DB_BACKEND=postgresql")
        return PostgreSQLVisionStore(db_url)
    return InfraredVisionStore(db_path)
