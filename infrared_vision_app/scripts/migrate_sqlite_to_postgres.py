from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from psycopg import connect

from db_store import PostgreSQLVisionStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate Infrared Vision data from SQLite to PostgreSQL."
    )
    parser.add_argument(
        "--sqlite-path",
        default=str(Path(__file__).resolve().parent.parent / "data" / "infrared_vision.db"),
        help="Path to the source SQLite database.",
    )
    parser.add_argument(
        "--postgres-dsn",
        required=True,
        help="PostgreSQL DSN. Example: postgresql://infrared:infrared@localhost:5432/infrared_vision",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate target PostgreSQL tables before importing data.",
    )
    return parser.parse_args()


def fetch_rows(sqlite_path: str, query: str) -> list[dict]:
    connection = sqlite3.connect(sqlite_path)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


class JsonModelAdapter:
    def __init__(self, payload: dict, **attrs) -> None:
        self._payload = payload
        for key, value in attrs.items():
            setattr(self, key, value)

    def model_dump(self, mode: str = "json") -> dict:  # noqa: ARG002
        return self._payload

    def dict(self) -> dict:
        return self._payload


def truncate_target_tables(postgres_dsn: str) -> None:
    with connect(postgres_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                TRUNCATE TABLE
                    infrared_vision.audit_logs,
                    infrared_vision.user_sessions,
                    infrared_vision.events,
                    infrared_vision.cameras,
                    infrared_vision.sites
                RESTART IDENTITY
                """
            )


def migrate_sites(store: PostgreSQLVisionStore, sqlite_path: str) -> int:
    rows = fetch_rows(sqlite_path, "SELECT payload FROM sites ORDER BY id")
    for row in rows:
        payload = row["payload"]
        payload_json = json.loads(payload) if isinstance(payload, str) else payload
        store.upsert_site(
            JsonModelAdapter(
                payload_json,
                id=payload_json["id"],
                profile=SimpleNamespace(value=payload_json["profile"]),
            )
        )
    return len(rows)


def migrate_cameras(store: PostgreSQLVisionStore, sqlite_path: str) -> int:
    rows = fetch_rows(sqlite_path, "SELECT payload FROM cameras ORDER BY id")
    for row in rows:
        payload = row["payload"]
        payload_json = json.loads(payload) if isinstance(payload, str) else payload
        store.upsert_camera(
            JsonModelAdapter(
                payload_json,
                id=payload_json["id"],
                site_id=payload_json["site_id"],
            )
        )
    return len(rows)


def migrate_events(store: PostgreSQLVisionStore, sqlite_path: str) -> int:
    rows = fetch_rows(sqlite_path, "SELECT payload FROM events ORDER BY occurred_at ASC")
    for row in rows:
        payload = row["payload"]
        payload_json = json.loads(payload) if isinstance(payload, str) else payload
        occurred_at = payload_json["occurred_at"]
        status_value = payload_json["status"]
        store.upsert_event(
            JsonModelAdapter(
                payload_json,
                id=payload_json["id"],
                site_id=payload_json["site_id"],
                camera_id=payload_json["camera_id"],
                occurred_at=SimpleNamespace(isoformat=lambda: occurred_at),
                status=SimpleNamespace(value=status_value),
            )
        )
    return len(rows)


def migrate_audit_logs(postgres_dsn: str, sqlite_path: str) -> int:
    rows = fetch_rows(sqlite_path, "SELECT action, actor, created_at, payload FROM audit_logs ORDER BY id ASC")
    with connect(postgres_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for row in rows:
                payload = json.loads(row["payload"]) if isinstance(row["payload"], str) else row["payload"]
                cur.execute(
                    """
                    INSERT INTO infrared_vision.audit_logs (action, actor, created_at, payload)
                    VALUES (%s, %s, %s, %s::jsonb)
                    """,
                    (row["action"], row["actor"], row["created_at"], json.dumps(payload, ensure_ascii=False)),
                )
    return len(rows)


def migrate_sessions(postgres_dsn: str, sqlite_path: str) -> int:
    rows = fetch_rows(
        sqlite_path,
        """
        SELECT session_id, username, user_role, refresh_token_hash,
               issued_at, expires_at, last_used_at, revoked_at, user_agent, ip_address
        FROM user_sessions
        ORDER BY issued_at ASC
        """,
    )
    with connect(postgres_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for row in rows:
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
                        row["session_id"],
                        row["username"],
                        row["user_role"],
                        row["refresh_token_hash"],
                        row["issued_at"],
                        row["expires_at"],
                        row["last_used_at"],
                        row["revoked_at"],
                        row["user_agent"],
                        row["ip_address"],
                    ),
                )
    return len(rows)


def main() -> None:
    args = parse_args()
    sqlite_path = str(Path(args.sqlite_path).expanduser().resolve())
    store = PostgreSQLVisionStore(args.postgres_dsn)

    if args.truncate:
        truncate_target_tables(args.postgres_dsn)

    site_count = migrate_sites(store, sqlite_path)
    camera_count = migrate_cameras(store, sqlite_path)
    event_count = migrate_events(store, sqlite_path)
    audit_count = migrate_audit_logs(args.postgres_dsn, sqlite_path)
    session_count = migrate_sessions(args.postgres_dsn, sqlite_path)

    print(
        json.dumps(
            {
                "sqlite_path": sqlite_path,
                "postgres_dsn": args.postgres_dsn,
                "sites": site_count,
                "cameras": camera_count,
                "events": event_count,
                "audit_logs": audit_count,
                "user_sessions": session_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
