from __future__ import annotations

from fastapi import HTTPException, status

from cv_client import CVInferenceClient
from models import (
    AppUser,
    CameraContext,
    CVInferenceRequest,
    Event,
    InferenceJobRequest,
    InferenceRequest,
    InferenceResponse,
    RiskLevel,
    UserRole,
)
from repository import InfraredVisionRepository
from vector_store import KnowledgeVectorService


class InfraredInferenceService:
    def __init__(
        self,
        repository: InfraredVisionRepository,
        vector_service: KnowledgeVectorService,
        cv_client: CVInferenceClient,
    ) -> None:
        self.repository = repository
        self.vector_service = vector_service
        self.cv_client = cv_client

    def run(self, payload: InferenceRequest, user: AppUser) -> InferenceResponse:
        return self._run_internal(
            InferenceJobRequest(
                profile=payload.profile,
                camera_id=payload.camera_id,
                thermal_summary=payload.thermal_summary,
                detected_objects=payload.detected_objects,
                thermal_matrix=payload.thermal_matrix,
                use_sliding_window=payload.use_sliding_window,
                sliding_window=payload.sliding_window,
                operator_note=payload.operator_note,
                auto_create_event=payload.auto_create_event,
            ),
            user,
        )

    def run_job_request(self, payload: InferenceJobRequest, user: AppUser) -> InferenceResponse:
        return self._run_internal(payload, user)

    def _run_internal(self, payload: InferenceJobRequest, user: AppUser) -> InferenceResponse:
        camera = self.repository.get_camera(payload.camera_id)
        if camera is None:
            raise HTTPException(status_code=404, detail="Camera not found")

        site = self.repository.get_site(camera.site_id)
        if site is None or site.profile != payload.profile:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="카메라와 프로필 정보가 일치하지 않습니다.",
            )

        cv_result = self.cv_client.infer(
            CVInferenceRequest(
                profile=payload.profile,
                camera=CameraContext(
                    id=camera.id,
                    zone=camera.zone,
                    stream_hint=camera.stream_hint,
                    objects_detected=camera.objects_detected,
                ),
                thermal_summary=payload.thermal_summary,
                detected_objects=payload.detected_objects,
                thermal_matrix=payload.thermal_matrix,
                use_sliding_window=payload.use_sliding_window,
                sliding_window=payload.sliding_window,
                operator_note=payload.operator_note,
            )
        )

        search_query = " ".join(
            [
                cv_result.risk_type,
                camera.zone,
                payload.thermal_summary,
                payload.operator_note or "",
            ]
        ).strip()
        guides = self.vector_service.search(
            search_query,
            profile=payload.profile,
            role=user.role,
            limit=3,
        )

        created_event: Event | None = None
        if payload.auto_create_event and cv_result.risk_level in {
            RiskLevel.MEDIUM,
            RiskLevel.HIGH,
            RiskLevel.CRITICAL,
        }:
            created_event = self.repository.create_event(
                site_id=camera.site_id,
                camera_id=camera.id,
                title=self._event_title(cv_result.risk_type, camera.zone),
                object_type=" / ".join(cv_result.object_labels) if cv_result.object_labels else "unknown",
                risk_type=cv_result.risk_type,
                risk_level=cv_result.risk_level,
                summary=cv_result.summary,
                recommended_action=cv_result.recommended_action,
                zone=camera.zone,
                distance_m=cv_result.distance_m,
                assignee=user.full_name if user.role != UserRole.VIEWER else None,
                created_by=f"{user.full_name} 추론 API",
            )

        return InferenceResponse(
            profile=payload.profile,
            camera=camera,
            detections=cv_result.detections,
            risk_level=cv_result.risk_level,
            risk_type=cv_result.risk_type,
            summary=cv_result.summary,
            recommended_action=cv_result.recommended_action,
            created_event=created_event,
            response_guides=guides.results,
            vector_backend=guides.backend,
            detection_strategy=cv_result.detection_strategy,
            scanned_windows=cv_result.scanned_windows,
            hotspot_count=cv_result.hotspot_count,
        )

    def _event_title(self, risk_type: str, zone: str) -> str:
        title_map = {
            "충돌주의": "작업자-지게차 근접 위험",
            "구역침범": "출입 제한구역 침범",
            "적재충돌주의": "적재물 주변 작업 혼선",
            "열점집중경고": "슬라이딩 윈도우 기반 열화상 이상 구역 감지",
            "국부과열주의": "슬라이딩 윈도우 기반 국부 고온 영역 감지",
            "보행자경고": "야간 보행자 위험 접근",
            "돌발장애물": "공사구간 돌발 장애물",
            "이상정차": "비정상 정차 차량 감지",
            "열점이상경고": "슬라이딩 윈도우 기반 도로 열점 이상 감지",
            "국부열점주의": "슬라이딩 윈도우 기반 단일 열점 감지",
            "이상상황": "열화상 이상행동 감지",
            "저시야경고": "저시야 구간 이상상황",
        }
        return title_map.get(risk_type, f"{zone} 위험 이벤트")
