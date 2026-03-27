from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles

from auth import AuthService
from auth_gateway import AuthGateway, ExternalIAMGateway, LocalAuthGateway
from cv_client import CVInferenceClient, LocalCVInferenceClient, RemoteCVInferenceClient
from inference_service import InfraredInferenceService
from job_queue import InferenceJobQueue, create_job_queue
from models import (
    AppUser,
    AuditLogEntry,
    AuthContext,
    DashboardPayload,
    ErrorResponse,
    ErrorInfo,
    Event,
    EventStatusUpdate,
    InferenceJobAccepted,
    InferenceJobEnvelope,
    InferenceJobRequest,
    InferenceRequest,
    InferenceResponse,
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    ResponseMeta,
    SiteProfile,
    SystemHealth,
    TokenRefreshRequest,
    UserRole,
    VectorSearchRequest,
    VectorSearchResponse,
    VectorStoreStatus,
)
from repository import InfraredVisionRepository
from service import InfraredVisionDashboardService
from settings import AppSettings, load_settings
from vector_store import KnowledgeVectorService


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
SECURITY = HTTPBearer(auto_error=False)


@dataclass
class AppContainer:
    settings: AppSettings
    repository: InfraredVisionRepository
    dashboard_service: InfraredVisionDashboardService
    auth_gateway: AuthGateway
    vector_service: KnowledgeVectorService
    cv_client: CVInferenceClient
    inference_service: InfraredInferenceService
    job_queue: InferenceJobQueue
    started_at: datetime


def build_auth_gateway(settings: AppSettings, repository: InfraredVisionRepository) -> AuthGateway:
    local_auth = AuthService(
        repository.store,
        secret=settings.auth_secret,
        access_expires_minutes=settings.auth_expires_minutes,
        refresh_expires_days=settings.refresh_expires_days,
    )
    if settings.iam_mode == "external":
        return ExternalIAMGateway(settings.iam_service_url)
    return LocalAuthGateway(local_auth)


def build_cv_client(settings: AppSettings) -> CVInferenceClient:
    if settings.cv_service_mode == "remote":
        return RemoteCVInferenceClient(settings.cv_service_url, settings.cv_service_timeout)
    return LocalCVInferenceClient()


def build_container(settings: AppSettings | None = None) -> AppContainer:
    resolved_settings = settings or load_settings(BASE_DIR)
    repository = InfraredVisionRepository(
        resolved_settings.db_path,
        db_backend=resolved_settings.db_backend,
        db_url=resolved_settings.db_url or None,
    )
    vector_service = KnowledgeVectorService(
        backend_name=resolved_settings.vector_db_backend,
        qdrant_url=resolved_settings.qdrant_url,
        qdrant_collection=resolved_settings.qdrant_collection,
        qdrant_timeout=resolved_settings.qdrant_timeout,
    )
    auth_gateway = build_auth_gateway(resolved_settings, repository)
    cv_client = build_cv_client(resolved_settings)
    dashboard_service = InfraredVisionDashboardService(repository)
    inference_service = InfraredInferenceService(repository, vector_service, cv_client)
    job_queue = create_job_queue(
        resolved_settings.queue_backend,
        resolved_settings.queue_url,
        resolved_settings.queue_name,
        resolved_settings.queue_result_ttl_seconds,
    )
    return AppContainer(
        settings=resolved_settings,
        repository=repository,
        dashboard_service=dashboard_service,
        auth_gateway=auth_gateway,
        vector_service=vector_service,
        cv_client=cv_client,
        inference_service=inference_service,
        job_queue=job_queue,
        started_at=datetime.now(timezone.utc).replace(microsecond=0),
    )


def get_container(request: Request) -> AppContainer:
    return request.app.state.container


def get_settings(container: AppContainer = Depends(get_container)) -> AppSettings:
    return container.settings


def get_repository(container: AppContainer = Depends(get_container)) -> InfraredVisionRepository:
    return container.repository


def get_dashboard_service(container: AppContainer = Depends(get_container)) -> InfraredVisionDashboardService:
    return container.dashboard_service


def get_auth_gateway(container: AppContainer = Depends(get_container)) -> AuthGateway:
    return container.auth_gateway


def get_vector_service(container: AppContainer = Depends(get_container)) -> KnowledgeVectorService:
    return container.vector_service


def get_inference_service(container: AppContainer = Depends(get_container)) -> InfraredInferenceService:
    return container.inference_service


def get_job_queue(container: AppContainer = Depends(get_container)) -> InferenceJobQueue:
    return container.job_queue


def _request_id_from(request: Request) -> str:
    return getattr(request.state, "request_id", request.headers.get("X-Request-ID", uuid4().hex))


def _dump_model(model) -> dict:
    dumper = getattr(model, "model_dump", None)
    if callable(dumper):
        return dumper(mode="json")
    return model.dict()


def _error_payload(
    request: Request,
    *,
    code: str,
    message: str,
    details: dict | list | str | None = None,
) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorInfo(code=code, message=message, details=details),
        meta=ResponseMeta(
            request_id=_request_id_from(request),
            timestamp=datetime.now(timezone.utc).replace(microsecond=0),
        ),
    )


def get_current_context(
    credentials: HTTPAuthorizationCredentials | None = Depends(SECURITY),
    auth_gateway: AuthGateway = Depends(get_auth_gateway),
) -> AuthContext:
    if credentials is None:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다.")
    return auth_gateway.verify_access_token(credentials.credentials)


def require_roles(*roles: UserRole):
    def dependency(context: AuthContext = Depends(get_current_context)) -> AuthContext:
        if context.user.role not in roles:
            raise HTTPException(status_code=403, detail="현재 계정으로는 수행할 수 없는 작업입니다.")
        return context

    return dependency


def create_app(settings: AppSettings | None = None) -> FastAPI:
    container = build_container(settings)
    app = FastAPI(
        title=container.settings.app_name,
        description="Distributed operations gateway for infrared camera monitoring.",
        version=container.settings.app_version,
    )
    app.state.container = container
    app.add_middleware(
        CORSMiddleware,
        allow_origins=container.settings.cors_allow_origins,
        allow_credentials=container.settings.cors_allow_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        payload = _error_payload(request, code=f"http_{exc.status_code}", message=str(exc.detail))
        return JSONResponse(status_code=exc.status_code, content=_dump_model(payload))

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        payload = _error_payload(
            request,
            code="validation_error",
            message="요청 본문 또는 파라미터 검증에 실패했습니다.",
            details=exc.errors(),
        )
        return JSONResponse(status_code=422, content=_dump_model(payload))

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
        payload = _error_payload(
            request,
            code="internal_server_error",
            message="예상하지 못한 서버 오류가 발생했습니다.",
        )
        return JSONResponse(status_code=500, content=_dump_model(payload))

    @app.middleware("http")
    async def add_request_context(request: Request, call_next):
        request.state.request_id = request.headers.get("X-Request-ID", uuid4().hex)
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["X-App-Version"] = app.state.container.settings.app_version
        return response

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health(settings: AppSettings = Depends(get_settings)) -> dict[str, str]:
        return {
            "status": "ok",
            "version": settings.app_version,
            "release_channel": settings.release_channel,
            "iam_mode": settings.iam_mode,
            "cv_service_mode": settings.cv_service_mode,
            "queue_backend": settings.queue_backend,
        }

    @app.get("/api/system/status", response_model=SystemHealth)
    def system_status(
        context: AuthContext = Depends(require_roles(UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN)),
        repository: InfraredVisionRepository = Depends(get_repository),
        vector_service: KnowledgeVectorService = Depends(get_vector_service),
        container: AppContainer = Depends(get_container),
    ) -> SystemHealth:
        _ = context
        stats = repository.stats()
        uptime_seconds = int((datetime.now(timezone.utc) - container.started_at).total_seconds())
        return SystemHealth(
            status="ok",
            app_version=container.settings.app_version,
            release_channel=container.settings.release_channel,
            database_path=str(stats["database_path"]),
            event_count=int(stats["event_count"]),
            audit_count=int(stats["audit_count"]),
            active_sessions=int(stats["active_sessions"]),
            vector_backend=vector_service.status().backend,
            started_at=container.started_at,
            uptime_seconds=uptime_seconds,
        )

    @app.post("/api/auth/login", response_model=LoginResponse)
    def login(
        payload: LoginRequest,
        request: Request,
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> LoginResponse:
        result = auth_gateway.authenticate(
            payload,
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None,
        )
        repository.log_audit_event(
            action="auth.login",
            actor=result.user.full_name,
            payload={"username": result.user.username, "role": result.user.role.value, "session_id": result.session_id},
        )
        return result

    @app.post("/api/auth/refresh", response_model=LoginResponse)
    def refresh_session(
        payload: TokenRefreshRequest,
        request: Request,
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> LoginResponse:
        result = auth_gateway.refresh(
            payload,
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None,
        )
        repository.log_audit_event(
            action="auth.refresh",
            actor=result.user.full_name,
            payload={"username": result.user.username, "role": result.user.role.value, "session_id": result.session_id},
        )
        return result

    @app.post("/api/auth/logout", response_model=LogoutResponse)
    def logout(
        context: AuthContext = Depends(get_current_context),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> LogoutResponse:
        auth_gateway.revoke_session(context.session_id)
        repository.log_audit_event(
            action="auth.logout",
            actor=context.user.full_name,
            payload={"username": context.user.username, "session_id": context.session_id},
        )
        return LogoutResponse(message="세션이 안전하게 종료되었습니다.")

    @app.get("/api/auth/me", response_model=AppUser)
    def auth_me(context: AuthContext = Depends(get_current_context)) -> AppUser:
        return context.user

    @app.get("/api/dashboard", response_model=DashboardPayload)
    def get_dashboard(
        profile: SiteProfile = Query(default=SiteProfile.LOGISTICS),
        context: AuthContext = Depends(require_roles(UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        service: InfraredVisionDashboardService = Depends(get_dashboard_service),
    ) -> DashboardPayload:
        auth_gateway.ensure_profile_access(context.user, profile)
        return service.build_dashboard(profile)

    @app.get("/api/events", response_model=list[Event])
    def get_events(
        profile: SiteProfile = Query(default=SiteProfile.LOGISTICS),
        context: AuthContext = Depends(require_roles(UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> list[Event]:
        auth_gateway.ensure_profile_access(context.user, profile)
        return repository.list_events(profile)

    @app.get("/api/events/{event_id}", response_model=Event)
    def get_event(
        event_id: str,
        context: AuthContext = Depends(require_roles(UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> Event:
        event = repository.get_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        site = repository.get_site(event.site_id)
        if site is None:
            raise HTTPException(status_code=404, detail="Event site not found")
        auth_gateway.ensure_profile_access(context.user, site.profile)
        return event

    @app.post("/api/events/{event_id}/acknowledge", response_model=Event)
    def acknowledge_event(
        event_id: str,
        context: AuthContext = Depends(require_roles(UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> Event:
        existing = repository.get_event(event_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Event not found")
        site = repository.get_site(existing.site_id)
        if site is None:
            raise HTTPException(status_code=404, detail="Event site not found")
        auth_gateway.ensure_profile_access(context.user, site.profile)
        event = repository.acknowledge_event(event_id)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        repository.log_audit_event(
            action="event.acknowledge",
            actor=context.user.full_name,
            payload={"event_id": event.id, "profile": site.profile.value},
        )
        return event

    @app.post("/api/events/{event_id}/status", response_model=Event)
    def update_event_status(
        event_id: str,
        payload: EventStatusUpdate,
        context: AuthContext = Depends(require_roles(UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> Event:
        existing = repository.get_event(event_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Event not found")
        site = repository.get_site(existing.site_id)
        if site is None:
            raise HTTPException(status_code=404, detail="Event site not found")
        auth_gateway.ensure_profile_access(context.user, site.profile)
        event = repository.update_event_status(event_id, payload.status, assignee=payload.assignee or context.user.full_name)
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        repository.log_audit_event(
            action="event.status_change",
            actor=context.user.full_name,
            payload={"event_id": event.id, "status": payload.status.value, "profile": site.profile.value},
        )
        return event

    @app.post("/api/inference/run", response_model=InferenceResponse)
    def run_inference(
        payload: InferenceRequest,
        context: AuthContext = Depends(require_roles(UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        inference_service: InfraredInferenceService = Depends(get_inference_service),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> InferenceResponse:
        auth_gateway.ensure_profile_access(context.user, payload.profile)
        result = inference_service.run(payload, context.user)
        repository.log_audit_event(
            action="inference.run",
            actor=context.user.full_name,
            payload={
                "camera_id": payload.camera_id,
                "profile": payload.profile.value,
                "risk_type": result.risk_type,
                "vector_backend": result.vector_backend,
                "created_event_id": result.created_event.id if result.created_event else None,
                "detection_strategy": result.detection_strategy,
            },
        )
        return result

    @app.post("/api/inference/jobs", response_model=InferenceJobAccepted)
    def enqueue_inference(
        payload: InferenceJobRequest,
        context: AuthContext = Depends(require_roles(UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        queue: InferenceJobQueue = Depends(get_job_queue),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> InferenceJobAccepted:
        auth_gateway.ensure_profile_access(context.user, payload.profile)
        accepted = queue.submit(payload, context.user)
        repository.log_audit_event(
            action="inference.job.enqueue",
            actor=context.user.full_name,
            payload={"job_id": accepted.job_id, "camera_id": payload.camera_id, "profile": payload.profile.value},
        )
        return accepted

    @app.get("/api/inference/jobs/{job_id}", response_model=InferenceJobEnvelope)
    def get_inference_job(
        job_id: str,
        context: AuthContext = Depends(require_roles(UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN)),
        queue: InferenceJobQueue = Depends(get_job_queue),
    ) -> InferenceJobEnvelope:
        _ = context
        envelope = queue.get(job_id)
        if envelope is None:
            raise HTTPException(status_code=404, detail="Inference job not found")
        return envelope

    @app.get("/api/vector/status", response_model=VectorStoreStatus)
    def vector_status(
        context: AuthContext = Depends(require_roles(UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN)),
        vector_service: KnowledgeVectorService = Depends(get_vector_service),
    ) -> VectorStoreStatus:
        _ = context
        return vector_service.status()

    @app.post("/api/vector/search", response_model=VectorSearchResponse)
    def vector_search(
        payload: VectorSearchRequest,
        context: AuthContext = Depends(require_roles(UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN)),
        auth_gateway: AuthGateway = Depends(get_auth_gateway),
        vector_service: KnowledgeVectorService = Depends(get_vector_service),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> VectorSearchResponse:
        if payload.profile is not None:
            auth_gateway.ensure_profile_access(context.user, payload.profile)
        result = vector_service.search(query=payload.query, profile=payload.profile, role=context.user.role, limit=payload.limit)
        repository.log_audit_event(
            action="vector.search",
            actor=context.user.full_name,
            payload={"query": payload.query, "profile": payload.profile.value if payload.profile else None, "backend": result.backend, "result_count": len(result.results)},
        )
        return result

    @app.get("/api/audit/logs", response_model=list[AuditLogEntry])
    def audit_logs(
        limit: int = Query(default=20, ge=1, le=100),
        context: AuthContext = Depends(require_roles(UserRole.ADMIN)),
        repository: InfraredVisionRepository = Depends(get_repository),
    ) -> list[AuditLogEntry]:
        _ = context
        return repository.list_audit_logs(limit=limit)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8010, reload=True)
