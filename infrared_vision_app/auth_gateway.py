from __future__ import annotations

import json
from urllib import request

from fastapi import HTTPException, status

from auth import AuthService
from models import AppUser, AuthContext, LoginRequest, LoginResponse, SiteProfile, TokenRefreshRequest


def _dump_model(model) -> dict:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    return model.dict()


class AuthGateway:
    def authenticate(
        self,
        payload: LoginRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        raise NotImplementedError

    def refresh(
        self,
        payload: TokenRefreshRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        raise NotImplementedError

    def revoke_session(self, session_id: str) -> None:
        raise NotImplementedError

    def verify_access_token(self, token: str) -> AuthContext:
        raise NotImplementedError

    def ensure_profile_access(self, user: AppUser, profile: SiteProfile) -> None:
        raise NotImplementedError


class LocalAuthGateway(AuthGateway):
    def __init__(self, service: AuthService) -> None:
        self.service = service

    def authenticate(self, payload: LoginRequest, *, user_agent: str | None = None, ip_address: str | None = None) -> LoginResponse:
        return self.service.authenticate(payload, user_agent=user_agent, ip_address=ip_address)

    def refresh(self, payload: TokenRefreshRequest, *, user_agent: str | None = None, ip_address: str | None = None) -> LoginResponse:
        return self.service.refresh(payload, user_agent=user_agent, ip_address=ip_address)

    def revoke_session(self, session_id: str) -> None:
        self.service.revoke_session(session_id)

    def verify_access_token(self, token: str) -> AuthContext:
        return self.service.verify_access_token(token)

    def ensure_profile_access(self, user: AppUser, profile: SiteProfile) -> None:
        self.service.ensure_profile_access(user, profile)


class ExternalIAMGateway(AuthGateway):
    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, path: str, *, method: str = "GET", payload: dict | None = None, headers: dict | None = None) -> dict:
        req = request.Request(
            f"{self.base_url}{path}",
            method=method,
            headers={"Content-Type": "application/json", **(headers or {})},
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def authenticate(
        self,
        payload: LoginRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        response = self._request(
            "/v1/auth/login",
            method="POST",
            payload=_dump_model(payload),
            headers={"X-Forwarded-For": ip_address or "", "User-Agent": user_agent or ""},
        )
        return LoginResponse(**response)

    def refresh(
        self,
        payload: TokenRefreshRequest,
        *,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> LoginResponse:
        response = self._request(
            "/v1/auth/refresh",
            method="POST",
            payload=_dump_model(payload),
            headers={"X-Forwarded-For": ip_address or "", "User-Agent": user_agent or ""},
        )
        return LoginResponse(**response)

    def revoke_session(self, session_id: str) -> None:
        self._request(f"/v1/auth/sessions/{session_id}", method="DELETE")

    def verify_access_token(self, token: str) -> AuthContext:
        response = self._request(
            "/v1/auth/introspect",
            method="POST",
            payload={"token": token},
        )
        return AuthContext(**response)

    def ensure_profile_access(self, user: AppUser, profile: SiteProfile) -> None:
        if profile not in user.allowed_profiles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="현재 계정은 해당 현장 프로필에 접근할 수 없습니다.",
            )
