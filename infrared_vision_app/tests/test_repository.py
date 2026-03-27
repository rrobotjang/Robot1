from __future__ import annotations

from pathlib import Path

from models import RiskLevel
from repository import InfraredVisionRepository


def test_create_event_generates_unique_ids(tmp_path: Path) -> None:
    repository = InfraredVisionRepository(tmp_path / "test.db")

    first = repository.create_event(
        site_id="site-logistics-a",
        camera_id="CAM-L-01",
        title="테스트 이벤트 1",
        object_type="worker",
        risk_type="충돌주의",
        risk_level=RiskLevel.HIGH,
        summary="요약 1",
        recommended_action="조치 1",
        zone="하역구역",
    )
    second = repository.create_event(
        site_id="site-logistics-a",
        camera_id="CAM-L-01",
        title="테스트 이벤트 2",
        object_type="worker",
        risk_type="충돌주의",
        risk_level=RiskLevel.HIGH,
        summary="요약 2",
        recommended_action="조치 2",
        zone="하역구역",
    )

    assert first.id != second.id
    assert repository.get_event(first.id) is not None
    assert repository.get_event(second.id) is not None


def test_stats_include_active_sessions(tmp_path: Path) -> None:
    repository = InfraredVisionRepository(tmp_path / "stats.db")
    stats = repository.stats()

    assert "active_sessions" in stats
    assert isinstance(stats["active_sessions"], int)
