from __future__ import annotations

from pathlib import Path

from auth import AuthService
from models import LoginRequest, TokenRefreshRequest
from storage import InfraredVisionStore


def test_refresh_rotates_refresh_token(tmp_path: Path) -> None:
    store = InfraredVisionStore(tmp_path / "auth.db")
    auth_service = AuthService(
        store,
        secret="test-secret",
        access_expires_minutes=30,
        refresh_expires_days=7,
    )

    login_result = auth_service.authenticate(LoginRequest(username="ops_admin", password="demo123!"))
    refresh_result = auth_service.refresh(
        TokenRefreshRequest(refresh_token=login_result.refresh_token)
    )

    assert refresh_result.session_id == login_result.session_id
    assert refresh_result.refresh_token != login_result.refresh_token
    assert refresh_result.access_token != login_result.access_token


def test_verify_access_token_rejects_revoked_session(tmp_path: Path) -> None:
    store = InfraredVisionStore(tmp_path / "revoked.db")
    auth_service = AuthService(
        store,
        secret="test-secret",
        access_expires_minutes=30,
        refresh_expires_days=7,
    )

    login_result = auth_service.authenticate(LoginRequest(username="ops_admin", password="demo123!"))
    auth_service.revoke_session(login_result.session_id)

    try:
        auth_service.verify_access_token(login_result.access_token)
    except Exception as exc:  # noqa: BLE001
        assert "세션이 유효하지 않습니다" in str(exc.detail)
    else:
        raise AssertionError("revoked session should not verify")
