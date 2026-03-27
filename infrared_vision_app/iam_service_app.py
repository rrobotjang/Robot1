from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Header, HTTPException

from auth import AuthService
from models import AuthContext, LoginRequest, LoginResponse, LogoutResponse, TokenRefreshRequest
from repository import InfraredVisionRepository
from settings import load_settings


BASE_DIR = Path(__file__).resolve().parent
SETTINGS = load_settings(BASE_DIR)
REPOSITORY = InfraredVisionRepository(
    SETTINGS.db_path,
    db_backend=SETTINGS.db_backend,
    db_url=SETTINGS.db_url or None,
)
AUTH_SERVICE = AuthService(
    REPOSITORY.store,
    secret=SETTINGS.auth_secret,
    access_expires_minutes=SETTINGS.auth_expires_minutes,
    refresh_expires_days=SETTINGS.refresh_expires_days,
)

app = FastAPI(title="Infrared External IAM", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "iam"}


@app.post("/v1/auth/login", response_model=LoginResponse)
def login(
    payload: LoginRequest,
    user_agent: str | None = Header(default=None),
    x_forwarded_for: str | None = Header(default=None),
) -> LoginResponse:
    return AUTH_SERVICE.authenticate(payload, user_agent=user_agent, ip_address=x_forwarded_for)


@app.post("/v1/auth/refresh", response_model=LoginResponse)
def refresh(
    payload: TokenRefreshRequest,
    user_agent: str | None = Header(default=None),
    x_forwarded_for: str | None = Header(default=None),
) -> LoginResponse:
    return AUTH_SERVICE.refresh(payload, user_agent=user_agent, ip_address=x_forwarded_for)


@app.post("/v1/auth/introspect", response_model=AuthContext)
def introspect(payload: dict[str, str]) -> AuthContext:
    token = payload.get("token")
    if not token:
        raise HTTPException(status_code=400, detail="token is required")
    return AUTH_SERVICE.verify_access_token(token)


@app.delete("/v1/auth/sessions/{session_id}", response_model=LogoutResponse)
def revoke_session(session_id: str) -> LogoutResponse:
    AUTH_SERVICE.revoke_session(session_id)
    return LogoutResponse(message="세션이 안전하게 종료되었습니다.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("iam_service_app:app", host="0.0.0.0", port=8020, reload=True)
