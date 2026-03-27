from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from db_store import VisionStoreProtocol, create_store
from models import AuditLogEntry, Camera, Event, EventStatus, RiskLevel, Site, SiteProfile, TimelineEntry
from sample_data import build_cameras, build_events, build_sites


def _copy_model(model, **update):
    copier = getattr(model, "model_copy", None)
    if callable(copier):
        return copier(update=update, deep=True)
    return model.copy(update=update, deep=True)


class InfraredVisionRepository:
    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        db_backend: str = "sqlite",
        db_url: str | None = None,
        store: VisionStoreProtocol | None = None,
    ) -> None:
        self.store = store or create_store(db_backend=db_backend, db_path=db_path, db_url=db_url)
        self.store.seed_if_empty(
            sites=build_sites(),
            cameras=build_cameras(),
            events=build_events(),
        )

    def list_sites(self, profile: SiteProfile | None = None) -> list[Site]:
        return [Site(**payload) for payload in self.store.fetch_sites(profile.value if profile else None)]

    def get_primary_site(self, profile: SiteProfile) -> Site:
        for site in self.list_sites(profile):
            if site.profile == profile:
                return site
        raise KeyError(f"Unknown profile: {profile}")

    def get_site(self, site_id: str) -> Site | None:
        payload = self.store.fetch_site(site_id)
        if payload is None:
            return None
        return Site(**payload)

    def list_cameras(self, profile: SiteProfile) -> list[Camera]:
        site_ids = {site.id for site in self.list_sites(profile)}
        cameras = [Camera(**payload) for payload in self.store.fetch_cameras(sorted(site_ids))]
        cameras.sort(key=lambda item: item.label)
        return cameras

    def get_camera(self, camera_id: str) -> Camera | None:
        payload = self.store.fetch_camera(camera_id)
        if payload is None:
            return None
        return Camera(**payload)

    def list_events(self, profile: SiteProfile) -> list[Event]:
        site_ids = {site.id for site in self.list_sites(profile)}
        events = [Event(**payload) for payload in self.store.fetch_events(sorted(site_ids))]
        events.sort(key=lambda item: item.occurred_at, reverse=True)
        return events

    def get_event(self, event_id: str) -> Event | None:
        payload = self.store.fetch_event(event_id)
        if payload is None:
            return None
        return Event(**payload)

    def acknowledge_event(self, event_id: str) -> Event | None:
        return self.update_event_status(
            event_id,
            EventStatus.ACKNOWLEDGED,
            title="이벤트 확인",
            detail="운영자가 이벤트를 확인하고 후속 조치를 준비했습니다.",
        )

    def update_event_status(
        self,
        event_id: str,
        status: EventStatus,
        assignee: str | None = None,
        title: str | None = None,
        detail: str | None = None,
    ) -> Event | None:
        event = self.get_event(event_id)
        if event is None:
            return None

        timeline = list(event.timeline)
        timeline.append(
            TimelineEntry(
                at=datetime.now().replace(microsecond=0),
                title=title or "상태 변경",
                detail=detail or f"이벤트 상태가 {status.value}로 변경되었습니다.",
            )
        )

        updated = _copy_model(
            event,
            status=status,
            assignee=assignee or event.assignee,
            timeline=timeline,
        )
        self.store.upsert_event(updated)
        return _copy_model(updated)

    def create_event(
        self,
        *,
        site_id: str,
        camera_id: str,
        title: str,
        object_type: str,
        risk_type: str,
        risk_level: RiskLevel,
        summary: str,
        recommended_action: str,
        zone: str,
        distance_m: float | None = None,
        assignee: str | None = None,
        created_by: str = "추론 API",
    ) -> Event:
        timestamp = datetime.now().replace(microsecond=0)
        event_id = f"EVT-AUTO-{timestamp:%y%m%d%H%M%S}"
        while self.get_event(event_id) is not None:
            event_id = f"EVT-AUTO-{timestamp:%y%m%d%H%M%S}-{uuid4().hex[:6].upper()}"

        event = Event(
            id=event_id,
            site_id=site_id,
            camera_id=camera_id,
            title=title,
            object_type=object_type,
            risk_type=risk_type,
            risk_level=risk_level,
            status=EventStatus.NEW,
            occurred_at=timestamp,
            summary=summary,
            recommended_action=recommended_action,
            zone=zone,
            distance_m=distance_m,
            assignee=assignee,
            timeline=[
                TimelineEntry(
                    at=timestamp,
                    title="이벤트 생성",
                    detail=f"{created_by}가 새로운 위험 이벤트를 생성했습니다.",
                )
            ],
        )
        self.store.upsert_event(event)
        return _copy_model(event)

    def log_audit_event(self, action: str, actor: str, payload: dict) -> None:
        self.store.insert_audit_log(
            action=action,
            actor=actor,
            payload=payload,
            created_at=datetime.now().replace(microsecond=0).isoformat(),
        )

    def list_audit_logs(self, limit: int = 50) -> list[AuditLogEntry]:
        rows = self.store.fetch_audit_logs(limit=limit)
        return [AuditLogEntry(**row) for row in rows]

    def stats(self) -> dict[str, int | str]:
        now_iso = datetime.now().replace(microsecond=0).isoformat()
        return {
            "database_path": str(self.store.path),
            "event_count": self.store.count_events(),
            "audit_count": self.store.count_audit_logs(),
            "active_sessions": self.store.count_active_sessions(now_iso),
        }
