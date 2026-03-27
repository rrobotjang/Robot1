from __future__ import annotations

from datetime import datetime, timedelta

from models import (
    AppUser,
    Camera,
    CameraStatus,
    Event,
    EventStatus,
    KnowledgeDocument,
    RiskLevel,
    Site,
    SiteProfile,
    TimelineEntry,
    UserRole,
)


def _minutes_ago(minutes: int) -> datetime:
    return datetime.now().replace(microsecond=0) - timedelta(minutes=minutes)


def build_sites() -> list[Site]:
    return [
        Site(
            id="site-logistics-a",
            name="물류센터 A동",
            profile=SiteProfile.LOGISTICS,
            location="인천 스마트 물류단지",
            zones=["하역구역", "지게차 동선", "보관구역", "출입제한구역"],
        ),
        Site(
            id="site-road-b",
            name="도로 관제 테스트베드 B",
            profile=SiteProfile.ROAD,
            location="판교 야간 도로 실증구간",
            zones=["공사구간", "보행자 통로", "정차금지 구역", "저시야 구간"],
        ),
    ]


def build_cameras() -> list[Camera]:
    return [
        Camera(
            id="CAM-L-01",
            site_id="site-logistics-a",
            label="하역구역 메인",
            zone="하역구역",
            status=CameraStatus.ONLINE,
            last_seen_at=_minutes_ago(1),
            stream_hint="사람 2명, 지게차 1대, 적재물 3건",
            objects_detected=["worker", "forklift", "pallet"],
        ),
        Camera(
            id="CAM-L-03",
            site_id="site-logistics-a",
            label="출입제한구역 측면",
            zone="출입제한구역",
            status=CameraStatus.DEGRADED,
            last_seen_at=_minutes_ago(4),
            stream_hint="적외선 대비 저하 감지, 작업자 1명 접근",
            objects_detected=["worker"],
        ),
        Camera(
            id="CAM-R-07",
            site_id="site-road-b",
            label="공사구간 북측",
            zone="공사구간",
            status=CameraStatus.ONLINE,
            last_seen_at=_minutes_ago(2),
            stream_hint="차량 4대, 공사장 장애물 1건",
            objects_detected=["car", "barrier"],
        ),
        Camera(
            id="CAM-R-09",
            site_id="site-road-b",
            label="보행자 통로 입구",
            zone="보행자 통로",
            status=CameraStatus.ONLINE,
            last_seen_at=_minutes_ago(1),
            stream_hint="보행자 3명, 자전거 1대",
            objects_detected=["pedestrian", "bicycle"],
        ),
    ]


def build_events() -> list[Event]:
    return [
        Event(
            id="EVT-LOG-240327-001",
            site_id="site-logistics-a",
            camera_id="CAM-L-01",
            title="작업자-지게차 근접 위험",
            object_type="worker / forklift",
            risk_type="충돌주의",
            risk_level=RiskLevel.CRITICAL,
            status=EventStatus.NEW,
            occurred_at=_minutes_ago(3),
            summary="하역구역에서 작업자와 지게차가 2.1m 이내로 접근했습니다.",
            recommended_action="현장 방송 후 지게차 속도 제한을 즉시 적용하세요.",
            zone="하역구역",
            distance_m=2.1,
            assignee="안전관리자 1",
            timeline=[
                TimelineEntry(
                    at=_minutes_ago(3),
                    title="이벤트 생성",
                    detail="적외선 카메라가 작업자와 지게차의 위험 접근을 감지했습니다.",
                ),
                TimelineEntry(
                    at=_minutes_ago(2),
                    title="위험 등급 상향",
                    detail="거리 기준이 임계값 미만으로 떨어져 CRITICAL로 상향했습니다.",
                ),
            ],
        ),
        Event(
            id="EVT-LOG-240327-002",
            site_id="site-logistics-a",
            camera_id="CAM-L-03",
            title="출입 제한구역 침범",
            object_type="worker",
            risk_type="구역침범",
            risk_level=RiskLevel.HIGH,
            status=EventStatus.ACKNOWLEDGED,
            occurred_at=_minutes_ago(9),
            summary="출입 제한구역에 작업자가 진입했습니다.",
            recommended_action="현장 호출 후 접근 금지선을 재안내하세요.",
            zone="출입제한구역",
            assignee="교대반장",
            timeline=[
                TimelineEntry(
                    at=_minutes_ago(9),
                    title="이벤트 생성",
                    detail="출입 제한구역 진입을 감지했습니다.",
                ),
                TimelineEntry(
                    at=_minutes_ago(7),
                    title="이벤트 확인",
                    detail="교대반장이 알림을 확인하고 현장 대응을 시작했습니다.",
                ),
            ],
        ),
        Event(
            id="EVT-ROAD-240327-001",
            site_id="site-road-b",
            camera_id="CAM-R-07",
            title="공사구간 돌발 장애물",
            object_type="road barrier",
            risk_type="돌발장애물",
            risk_level=RiskLevel.HIGH,
            status=EventStatus.IN_PROGRESS,
            occurred_at=_minutes_ago(5),
            summary="공사구간 북측 차선에 이동식 장애물이 진입했습니다.",
            recommended_action="차선 통제 알림을 송출하고 관리 차량을 출동시키세요.",
            zone="공사구간",
            assignee="도로관제팀",
            timeline=[
                TimelineEntry(
                    at=_minutes_ago(5),
                    title="이벤트 생성",
                    detail="장애물 객체가 주행 차선에 진입했습니다.",
                ),
                TimelineEntry(
                    at=_minutes_ago(4),
                    title="관제팀 배정",
                    detail="관제팀에 출동 요청을 전달했습니다.",
                ),
            ],
        ),
        Event(
            id="EVT-ROAD-240327-002",
            site_id="site-road-b",
            camera_id="CAM-R-09",
            title="야간 보행자 위험 접근",
            object_type="pedestrian",
            risk_type="보행자경고",
            risk_level=RiskLevel.MEDIUM,
            status=EventStatus.NEW,
            occurred_at=_minutes_ago(1),
            summary="보행자가 저시야 구간 가장자리로 접근했습니다.",
            recommended_action="점멸 경고등과 방송 알림을 활성화하세요.",
            zone="보행자 통로",
            assignee="야간 운영자",
            timeline=[
                TimelineEntry(
                    at=_minutes_ago(1),
                    title="이벤트 생성",
                    detail="적외선 카메라가 보행자 접근을 탐지했습니다.",
                ),
            ],
        ),
    ]


def build_demo_users() -> list[AppUser]:
    return [
        AppUser(
            username="ops_admin",
            full_name="통합 운영 관리자",
            role=UserRole.ADMIN,
            allowed_profiles=[SiteProfile.LOGISTICS, SiteProfile.ROAD],
        ),
        AppUser(
            username="field_operator",
            full_name="물류 운영자",
            role=UserRole.OPERATOR,
            allowed_profiles=[SiteProfile.LOGISTICS],
        ),
        AppUser(
            username="road_manager",
            full_name="도로 관제 운영자",
            role=UserRole.OPERATOR,
            allowed_profiles=[SiteProfile.ROAD],
        ),
        AppUser(
            username="safety_viewer",
            full_name="안전 모니터링 조회자",
            role=UserRole.VIEWER,
            allowed_profiles=[SiteProfile.LOGISTICS, SiteProfile.ROAD],
        ),
    ]


def build_knowledge_documents() -> list[KnowledgeDocument]:
    return [
        KnowledgeDocument(
            id="kb-logistics-collision-playbook",
            title="물류센터 작업자-지게차 충돌 대응 가이드",
            body=(
                "작업자와 지게차가 3m 이내로 접근하면 현장 경고 방송을 즉시 송출하고 "
                "지게차 속도 제한 모드를 적용한다. 작업자를 안전 구역으로 유도한 뒤 "
                "교대반장이 재발 방지 확인을 완료해야 한다."
            ),
            tags=["logistics", "forklift", "worker", "collision", "playbook"],
            profile=SiteProfile.LOGISTICS,
            allowed_roles=[UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN],
        ),
        KnowledgeDocument(
            id="kb-logistics-zone-intrusion",
            title="출입 제한구역 침범 대응 절차",
            body=(
                "출입 제한구역 침범이 발생하면 해당 카메라의 경광등을 켜고 접근 금지선을 재안내한다. "
                "반복 침범 구역은 현장 맵에서 고위험 구간으로 승격해 야간 순찰 빈도를 높인다."
            ),
            tags=["logistics", "zone", "intrusion", "safety"],
            profile=SiteProfile.LOGISTICS,
            allowed_roles=[UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN],
        ),
        KnowledgeDocument(
            id="kb-road-pedestrian-night",
            title="야간 보행자 위험 경보 운영 기준",
            body=(
                "야간 보행자 경보는 저시야 구간에서 차량과 보행자 이동 벡터가 교차할 때 우선 발령한다. "
                "점멸 경고등, VMS 표출, 제한 속도 하향을 순차적으로 적용한다."
            ),
            tags=["road", "pedestrian", "night", "warning"],
            profile=SiteProfile.ROAD,
            allowed_roles=[UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN],
        ),
        KnowledgeDocument(
            id="kb-road-obstacle-response",
            title="공사구간 돌발 장애물 대응 플레이북",
            body=(
                "공사구간 주행 차선에 장애물이 탐지되면 차선 통제 알림을 즉시 송출하고 "
                "현장 관리 차량을 출동시킨다. 장애물 제거 전까지 우회 차선과 제한 속도를 유지한다."
            ),
            tags=["road", "barrier", "incident", "playbook"],
            profile=SiteProfile.ROAD,
            allowed_roles=[UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN],
        ),
        KnowledgeDocument(
            id="kb-admin-policy-template",
            title="위험도 정책 조정 체크리스트",
            body=(
                "관리자는 카메라 감도, 경고 거리, 조치 SLA, 보존 기간을 주간 단위로 검토해야 한다. "
                "오탐 증가 시 임계값과 금지 구역 ROI를 재조정하고 변경 이력을 감사 로그에 남긴다."
            ),
            tags=["admin", "policy", "audit", "threshold"],
            profile=None,
            allowed_roles=[UserRole.ADMIN],
        ),
        KnowledgeDocument(
            id="kb-common-thermal-calibration",
            title="적외선 카메라 열화상 보정 체크포인트",
            body=(
                "적외선 카메라의 대비가 저하되면 렌즈 오염, 열화상 범위 설정, 야간 반사체 여부를 우선 점검한다. "
                "보정 후 동일 구역에서 재추론을 수행해 오탐률을 비교한다."
            ),
            tags=["thermal", "calibration", "camera", "common"],
            profile=None,
            allowed_roles=[UserRole.VIEWER, UserRole.OPERATOR, UserRole.ADMIN],
        ),
    ]
