from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app import create_app
from settings import AppSettings


def build_test_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        app_name="Infrared Vision Test App",
        app_version="test",
        release_channel="test",
        cors_allow_origins=["*"],
        auth_secret="test-secret",
        auth_expires_minutes=30,
        refresh_expires_days=7,
        vector_db_backend="memory",
        qdrant_url="http://127.0.0.1:6333",
        qdrant_collection="infrared_vision_knowledge",
        qdrant_timeout=1.0,
        db_backend="sqlite",
        db_url="",
        db_path=tmp_path / "api-test.db",
        auto_refresh_seconds=20,
        iam_mode="local",
        iam_service_url="http://127.0.0.1:8020",
        cv_service_mode="local",
        cv_service_url="http://127.0.0.1:8030",
        cv_service_timeout=1.0,
        queue_backend="memory",
        queue_url="redis://127.0.0.1:6379/0",
        queue_name="infrared:test:queue",
        queue_result_ttl_seconds=300,
    )


def test_login_dashboard_and_refresh_flow(tmp_path: Path) -> None:
    app = create_app(build_test_settings(tmp_path))
    client = TestClient(app)

    login_response = client.post(
        "/api/auth/login",
        json={"username": "ops_admin", "password": "demo123!"},
    )
    assert login_response.status_code == 200

    tokens = login_response.json()
    access_token = tokens["access_token"]
    refresh_token = tokens["refresh_token"]

    dashboard_response = client.get(
        "/api/dashboard?profile=logistics",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert dashboard_response.status_code == 200
    assert dashboard_response.json()["profile"] == "logistics"

    refresh_response = client.post(
        "/api/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_response.status_code == 200
    assert refresh_response.json()["refresh_token"] != refresh_token


def test_logout_invalidates_existing_access_token(tmp_path: Path) -> None:
    app = create_app(build_test_settings(tmp_path))
    client = TestClient(app)

    login_response = client.post(
        "/api/auth/login",
        json={"username": "ops_admin", "password": "demo123!"},
    )
    access_token = login_response.json()["access_token"]

    logout_response = client.post(
        "/api/auth/logout",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert logout_response.status_code == 200

    auth_me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert auth_me_response.status_code == 401


def test_inference_supports_sliding_window_scan(tmp_path: Path) -> None:
    app = create_app(build_test_settings(tmp_path))
    client = TestClient(app)

    login_response = client.post(
        "/api/auth/login",
        json={"username": "ops_admin", "password": "demo123!"},
    )
    access_token = login_response.json()["access_token"]

    inference_response = client.post(
        "/api/inference/run",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "profile": "logistics",
            "camera_id": "CAM-L-01",
            "thermal_summary": "하역구역 일부 구간에서 국부 고온 영역이 보임",
            "detected_objects": [],
            "use_sliding_window": True,
            "thermal_matrix": [
                [0.12, 0.15, 0.20, 0.18],
                [0.18, 0.82, 0.88, 0.24],
                [0.20, 0.84, 0.91, 0.26],
                [0.12, 0.18, 0.22, 0.15],
            ],
            "auto_create_event": False,
        },
    )

    assert inference_response.status_code == 200
    body = inference_response.json()
    assert body["detection_strategy"] == "sliding_window_hybrid"
    assert body["scanned_windows"] > 0
    assert body["hotspot_count"] >= 1


def test_can_enqueue_inference_job(tmp_path: Path) -> None:
    app = create_app(build_test_settings(tmp_path))
    client = TestClient(app)

    login_response = client.post(
        "/api/auth/login",
        json={"username": "ops_admin", "password": "demo123!"},
    )
    access_token = login_response.json()["access_token"]

    enqueue_response = client.post(
        "/api/inference/jobs",
        headers={"Authorization": f"Bearer {access_token}"},
        json={
            "profile": "logistics",
            "camera_id": "CAM-L-01",
            "thermal_summary": "작업자와 지게차가 접근 중",
            "detected_objects": ["worker", "forklift"],
        },
    )

    assert enqueue_response.status_code == 200
    job_id = enqueue_response.json()["job_id"]

    job_response = client.get(
        f"/api/inference/jobs/{job_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert job_response.status_code == 200
    assert job_response.json()["status"] == "queued"
