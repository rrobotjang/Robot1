from __future__ import annotations

from datetime import datetime

from models import CameraStatus, DashboardPayload, EventStatus, RiskLevel, SiteProfile, SummaryCard
from repository import InfraredVisionRepository


class InfraredVisionDashboardService:
    def __init__(self, repository: InfraredVisionRepository) -> None:
        self.repository = repository

    def build_dashboard(self, profile: SiteProfile) -> DashboardPayload:
        site = self.repository.get_primary_site(profile)
        cameras = self.repository.list_cameras(profile)
        events = self.repository.list_events(profile)

        online = sum(1 for camera in cameras if camera.status == CameraStatus.ONLINE)
        degraded = sum(1 for camera in cameras if camera.status == CameraStatus.DEGRADED)
        open_events = sum(1 for event in events if event.status != EventStatus.RESOLVED)
        critical = sum(1 for event in events if event.risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL})

        summary = [
            SummaryCard(
                id="active-cameras",
                label="가동 카메라",
                value=f"{online}/{len(cameras)}",
                caption="실시간 스트림 수집 중",
                tone="info",
            ),
            SummaryCard(
                id="attention-cameras",
                label="주의 필요 장비",
                value=str(degraded),
                caption="화질 또는 통신 상태 점검 필요",
                tone="warning",
            ),
            SummaryCard(
                id="open-events",
                label="미해결 이벤트",
                value=str(open_events),
                caption="운영자 조치가 필요한 건수",
                tone="neutral",
            ),
            SummaryCard(
                id="high-risk",
                label="고위험 경보",
                value=str(critical),
                caption="HIGH 이상 등급",
                tone="danger",
            ),
        ]

        selected_event_id = events[0].id if events else None
        return DashboardPayload(
            profile=profile,
            site=site,
            summary=summary,
            cameras=cameras,
            events=events,
            selected_event_id=selected_event_id,
            generated_at=datetime.now().replace(microsecond=0),
        )
