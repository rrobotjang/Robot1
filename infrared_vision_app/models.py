from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class SiteProfile(str, Enum):
    LOGISTICS = "logistics"
    ROAD = "road"


class CameraStatus(str, Enum):
    ONLINE = "online"
    DEGRADED = "degraded"
    OFFLINE = "offline"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventStatus(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"


class UserRole(str, Enum):
    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"


class Site(BaseModel):
    id: str
    name: str
    profile: SiteProfile
    location: str
    zones: List[str]


class Camera(BaseModel):
    id: str
    site_id: str
    label: str
    zone: str
    status: CameraStatus
    last_seen_at: datetime
    stream_hint: str
    objects_detected: List[str] = Field(default_factory=list)


class TimelineEntry(BaseModel):
    at: datetime
    title: str
    detail: str


class Event(BaseModel):
    id: str
    site_id: str
    camera_id: str
    title: str
    object_type: str
    risk_type: str
    risk_level: RiskLevel
    status: EventStatus
    occurred_at: datetime
    summary: str
    recommended_action: str
    zone: str
    distance_m: Optional[float] = None
    assignee: Optional[str] = None
    timeline: List[TimelineEntry] = Field(default_factory=list)


class SummaryCard(BaseModel):
    id: str
    label: str
    value: str
    caption: str
    tone: str


class DashboardPayload(BaseModel):
    profile: SiteProfile
    site: Site
    summary: List[SummaryCard]
    cameras: List[Camera]
    events: List[Event]
    selected_event_id: Optional[str] = None
    generated_at: datetime


class EventStatusUpdate(BaseModel):
    status: EventStatus
    assignee: Optional[str] = None


class AppUser(BaseModel):
    username: str
    full_name: str
    role: UserRole
    allowed_profiles: List[SiteProfile] = Field(default_factory=list)


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    session_id: str
    token_type: str = "bearer"
    expires_at: datetime
    user: AppUser


class TokenRefreshRequest(BaseModel):
    refresh_token: str


class LogoutResponse(BaseModel):
    message: str


class AuthContext(BaseModel):
    user: AppUser
    session_id: str
    expires_at: datetime


class AuditLogEntry(BaseModel):
    id: int
    action: str
    actor: str
    created_at: datetime
    payload: dict = Field(default_factory=dict)


class SystemHealth(BaseModel):
    status: str
    app_version: str
    release_channel: str
    database_path: str
    event_count: int
    audit_count: int
    active_sessions: int
    vector_backend: str
    started_at: datetime
    uptime_seconds: int


class BoundingBox(BaseModel):
    left: float
    top: float
    width: float
    height: float


class SlidingWindowConfig(BaseModel):
    window_sizes: List[int] = Field(default_factory=lambda: [2, 3, 4])
    stride: int = 1
    heat_threshold: float = 0.72
    min_hot_cells: int = 2
    max_windows: int = 6


class InferenceDetection(BaseModel):
    label: str
    confidence: float
    heat_score: float
    bbox: Optional[BoundingBox] = None


class CameraContext(BaseModel):
    id: str
    zone: str
    stream_hint: str
    objects_detected: List[str] = Field(default_factory=list)


class KnowledgeDocument(BaseModel):
    id: str
    title: str
    body: str
    tags: List[str] = Field(default_factory=list)
    profile: Optional[SiteProfile] = None
    allowed_roles: List[UserRole] = Field(default_factory=list)


class VectorSearchRequest(BaseModel):
    query: str
    profile: Optional[SiteProfile] = None
    limit: int = 4


class VectorSearchResult(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    tags: List[str] = Field(default_factory=list)
    profile: Optional[SiteProfile] = None


class VectorSearchResponse(BaseModel):
    backend: str
    query: str
    results: List[VectorSearchResult] = Field(default_factory=list)


class VectorStoreStatus(BaseModel):
    backend: str
    document_count: int


class InferenceRequest(BaseModel):
    profile: SiteProfile
    camera_id: str
    thermal_summary: str
    detected_objects: List[str] = Field(default_factory=list)
    thermal_matrix: List[List[float]] = Field(default_factory=list)
    use_sliding_window: bool = False
    sliding_window: SlidingWindowConfig = Field(default_factory=SlidingWindowConfig)
    operator_note: Optional[str] = None
    auto_create_event: bool = True


class CVInferenceRequest(BaseModel):
    profile: SiteProfile
    camera: CameraContext
    thermal_summary: str
    detected_objects: List[str] = Field(default_factory=list)
    thermal_matrix: List[List[float]] = Field(default_factory=list)
    use_sliding_window: bool = False
    sliding_window: SlidingWindowConfig = Field(default_factory=SlidingWindowConfig)
    operator_note: Optional[str] = None


class CVInferenceResponse(BaseModel):
    detections: List[InferenceDetection] = Field(default_factory=list)
    risk_level: RiskLevel
    risk_type: str
    summary: str
    recommended_action: str
    distance_m: Optional[float] = None
    object_labels: List[str] = Field(default_factory=list)
    detection_strategy: str = "semantic"
    scanned_windows: int = 0
    hotspot_count: int = 0


class InferenceResponse(BaseModel):
    profile: SiteProfile
    camera: Camera
    detections: List[InferenceDetection] = Field(default_factory=list)
    risk_level: RiskLevel
    risk_type: str
    summary: str
    recommended_action: str
    created_event: Optional[Event] = None
    response_guides: List[VectorSearchResult] = Field(default_factory=list)
    vector_backend: str
    detection_strategy: str = "semantic"
    scanned_windows: int = 0
    hotspot_count: int = 0


class InferenceJobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class InferenceJobRequest(BaseModel):
    profile: SiteProfile
    camera_id: str
    thermal_summary: str
    detected_objects: List[str] = Field(default_factory=list)
    thermal_matrix: List[List[float]] = Field(default_factory=list)
    use_sliding_window: bool = False
    sliding_window: SlidingWindowConfig = Field(default_factory=SlidingWindowConfig)
    operator_note: Optional[str] = None
    auto_create_event: bool = True


class InferenceJobEnvelope(BaseModel):
    job_id: str
    status: InferenceJobStatus
    request: InferenceJobRequest
    requested_by: AppUser
    queued_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[InferenceResponse] = None
    error: Optional[str] = None


class InferenceJobAccepted(BaseModel):
    job_id: str
    status: InferenceJobStatus
    queued_at: datetime


class ResponseMeta(BaseModel):
    request_id: str
    timestamp: datetime


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: Optional[dict | list | str] = None


class ErrorResponse(BaseModel):
    error: ErrorInfo
    meta: ResponseMeta
