from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from fastapi import HTTPException, status

from models import (
    AppUser,
    AuthContext,
    LoginRequest,
    LoginResponse,
    SiteProfile,
    TokenRefreshRequest,
)
from sample_data import build_demo_users
from storage import InfraredVisionStore


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value)


class AuthService:
    def __init__(
        self,
        store: InfraredVisionStore,
        *,
        secret: str,
        access_expires_minutes: int,
        refresh_expires_days: int,
    ) -> None:
        self.store = store
        self.secret = secret.encode("utf-8")
        self.expires_minutes = access_expires_minutes
        self.refresh_expires_days = refresh_expires_days
        self._user_map = {user.username: user for user in build_demo_users()}
        self._password_hashes = {
            username: self._hash_password(username, "demo123!")
            for username in self._user_map
        }

    def _hash_password(self, username: str, raw_password: str) -> str:
        salt = f"infrared-vision::{username}".encode("utf-8")
        return hashlib.pbkdf2_hmac(
            "sha256",
            raw_password.encode("utf-8"),
            salt,
            120_000,
        ).hex()

    def _hash_refresh_token(self, token: str) -> str:
        return hmac.new(self.secret, token.encode("utf-8"), hashlib.sha256).hexdigest()

    def _b64encode(self, raw: bytes) -> str:
        return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("utf-8")

    def _b64decode(self, raw: str) -> bytes:
        padding = "=" * (-len(raw) % 4)
        return base64.urlsafe_b64decode(raw + padding)

    def _parse_refresh_token(self, token: str) -> str:
        session_id, separator, _secret = token.partition(".")
        if not separator or not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="리프레시 토큰 형식이 올바르지 않습니다.",
            )
        return session_id

    def _issue_access_token(self, user: AppUser, session_id: str, expires_at: datetime) -> str:
        payload = {
            "sub": user.username,
            "name": user.full_name,
            "role": user.role.value,
            "profiles": [profile.value for profile in user.allowed_profiles],
            "sid": session_id,
            "typ": "access",
            "iat": int(_utc_now().timestamp()),
            "jti": uuid4().hex,
            "exp": int(expires_at.timestamp()),
        }
        encoded_payload = self._b64encode(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )
        signature = hmac.new(
            self.secret,
            encoded_payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        return f"{encoded_payload}.{self._b64encode(signature)}"

    def _issue_refresh_token(self, session_id: str) -> str:
        return f"{session_id}.{secrets.token_urlsafe(32)}"

    def _resolve_user(self, username: str) -> AppUser:
        user = self._user_map.get(username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="인증된 사용자 정보를 찾을 수 없습니다.",
            )
        return user

    def _session_active(self, session: dict | None) -> bool:
        if session is None:
            return False
        if session.get("revoked_at"):
            return False
        expires_at = _coerce_datetime(session["expires_at"])
        return expires_at >= _utc_now()

    def _build_login_response(
        self,
        *,
        user: AppUser,
        session_id: str,
        refresh_token: str,
        access_expires_at: datetime,
    ) -> LoginResponse:
        access_token = self._issue_access_token(user, session_id, access_expires_at)
        return LoginResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            session_id=session_id,
            expires_at=access_expires_at,
            user=user,
        )

    def authenticate(
        self,
        payload: LoginRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        user = self._user_map.get(payload.username)
        password_hash = self._password_hashes.get(payload.username)

        if user is None or password_hash != self._hash_password(payload.username, payload.password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="아이디 또는 비밀번호가 올바르지 않습니다.",
            )

        issued_at = _utc_now()
        access_expires_at = issued_at + timedelta(minutes=self.expires_minutes)
        refresh_expires_at = issued_at + timedelta(days=self.refresh_expires_days)
        session_id = uuid4().hex
        refresh_token = self._issue_refresh_token(session_id)

        self.store.upsert_session(
            session_id=session_id,
            username=user.username,
            user_role=user.role.value,
            refresh_token_hash=self._hash_refresh_token(refresh_token),
            issued_at=issued_at.isoformat(),
            expires_at=refresh_expires_at.isoformat(),
            last_used_at=issued_at.isoformat(),
            user_agent=user_agent,
            ip_address=ip_address,
        )

        return self._build_login_response(
            user=user,
            session_id=session_id,
            refresh_token=refresh_token,
            access_expires_at=access_expires_at,
        )

    def refresh(
        self,
        payload: TokenRefreshRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        session_id = self._parse_refresh_token(payload.refresh_token)
        session = self.store.fetch_session(session_id)

        if not self._session_active(session):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="세션이 만료되었거나 이미 종료되었습니다.",
            )

        expected_hash = session["refresh_token_hash"]
        provided_hash = self._hash_refresh_token(payload.refresh_token)
        if not hmac.compare_digest(expected_hash, provided_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="리프레시 토큰 검증에 실패했습니다.",
            )

        user = self._resolve_user(session["username"])
        now = _utc_now()
        access_expires_at = now + timedelta(minutes=self.expires_minutes)
        refresh_expires_at = now + timedelta(days=self.refresh_expires_days)
        rotated_refresh_token = self._issue_refresh_token(session_id)

        self.store.upsert_session(
            session_id=session_id,
            username=user.username,
            user_role=user.role.value,
            refresh_token_hash=self._hash_refresh_token(rotated_refresh_token),
            issued_at=session["issued_at"],
            expires_at=refresh_expires_at.isoformat(),
            last_used_at=now.isoformat(),
            user_agent=user_agent or session.get("user_agent"),
            ip_address=ip_address or session.get("ip_address"),
            revoked_at=None,
        )

        return self._build_login_response(
            user=user,
            session_id=session_id,
            refresh_token=rotated_refresh_token,
            access_expires_at=access_expires_at,
        )

    def revoke_session(self, session_id: str) -> None:
        self.store.revoke_session(session_id, _utc_now().isoformat())

    def verify_access_token(self, token: str) -> AuthContext:
        try:
            encoded_payload, encoded_signature = token.split(".", 1)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="인증 토큰 형식이 올바르지 않습니다.",
            ) from exc

        expected_signature = hmac.new(
            self.secret,
            encoded_payload.encode("utf-8"),
            hashlib.sha256,
        ).digest()

        if not hmac.compare_digest(self._b64encode(expected_signature), encoded_signature):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="인증 토큰 검증에 실패했습니다.",
            )

        payload = json.loads(self._b64decode(encoded_payload).decode("utf-8"))
        if payload.get("typ") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="허용되지 않은 토큰 유형입니다.",
            )

        if int(payload["exp"]) < int(_utc_now().timestamp()):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="인증 토큰이 만료되었습니다.",
            )

        session = self.store.fetch_session(payload["sid"])
        if not self._session_active(session):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="세션이 유효하지 않습니다. 다시 로그인해주세요.",
            )

        user = self._resolve_user(payload["sub"])
        return AuthContext(
            user=user,
            session_id=payload["sid"],
            expires_at=datetime.fromtimestamp(int(payload["exp"]), tz=timezone.utc),
        )

    def verify_token(self, token: str) -> AppUser:
        return self.verify_access_token(token).user

    def ensure_profile_access(self, user: AppUser, profile: SiteProfile) -> None:
        if profile not in user.allowed_profiles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="현재 계정은 해당 현장 프로필에 접근할 수 없습니다.",
            )
