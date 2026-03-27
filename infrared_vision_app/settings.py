from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_origins(raw: str) -> list[str]:
    origins = [item.strip() for item in raw.split(",") if item.strip()]
    return origins or ["*"]


@dataclass(frozen=True)
class AppSettings:
    app_name: str
    app_version: str
    release_channel: str
    cors_allow_origins: list[str]
    auth_secret: str
    auth_expires_minutes: int
    refresh_expires_days: int
    vector_db_backend: str
    qdrant_url: str
    qdrant_collection: str
    qdrant_timeout: float
    db_backend: str
    db_url: str
    db_path: Path
    auto_refresh_seconds: int
    iam_mode: str
    iam_service_url: str
    cv_service_mode: str
    cv_service_url: str
    cv_service_timeout: float
    queue_backend: str
    queue_url: str
    queue_name: str
    queue_result_ttl_seconds: int


def load_settings(base_dir: Path) -> AppSettings:
    default_db_path = base_dir / "data" / "infrared_vision.db"
    return AppSettings(
        app_name=os.getenv("APP_NAME", "Infrared Vision Safety App"),
        app_version=os.getenv("APP_VERSION", "0.2.0"),
        release_channel=os.getenv("APP_RELEASE_CHANNEL", "production-ready-demo"),
        cors_allow_origins=_parse_origins(os.getenv("CORS_ALLOW_ORIGINS", "*")),
        auth_secret=os.getenv("AUTH_SECRET", "infrared-vision-demo-secret"),
        auth_expires_minutes=int(os.getenv("AUTH_EXPIRES_MINUTES", "30")),
        refresh_expires_days=int(os.getenv("AUTH_REFRESH_EXPIRES_DAYS", "7")),
        vector_db_backend=os.getenv("VECTOR_DB_BACKEND", "memory").lower(),
        qdrant_url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        qdrant_collection=os.getenv("QDRANT_COLLECTION", "infrared_vision_knowledge"),
        qdrant_timeout=float(os.getenv("QDRANT_TIMEOUT", "3.0")),
        db_backend=os.getenv("DB_BACKEND", "sqlite").lower(),
        db_url=os.getenv("DB_URL", ""),
        db_path=Path(os.getenv("APP_DB_PATH", default_db_path)),
        auto_refresh_seconds=int(os.getenv("AUTO_REFRESH_SECONDS", "20")),
        iam_mode=os.getenv("IAM_MODE", "local").lower(),
        iam_service_url=os.getenv("IAM_SERVICE_URL", "http://infrared_iam:8020"),
        cv_service_mode=os.getenv("CV_SERVICE_MODE", "local").lower(),
        cv_service_url=os.getenv("CV_SERVICE_URL", "http://infrared_cv_service:8030"),
        cv_service_timeout=float(os.getenv("CV_SERVICE_TIMEOUT", "10.0")),
        queue_backend=os.getenv("QUEUE_BACKEND", "memory").lower(),
        queue_url=os.getenv("QUEUE_URL", "redis://redis:6379/0"),
        queue_name=os.getenv("QUEUE_NAME", "infrared:inference:queue"),
        queue_result_ttl_seconds=int(os.getenv("QUEUE_RESULT_TTL_SECONDS", "3600")),
    )
