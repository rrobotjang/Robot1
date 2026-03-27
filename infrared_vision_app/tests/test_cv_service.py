from __future__ import annotations

from fastapi.testclient import TestClient

from cv_service_app import app


def test_cv_service_infer_endpoint() -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/infer",
        json={
            "profile": "logistics",
            "camera": {
                "id": "CAM-L-01",
                "zone": "하역구역",
                "stream_hint": "사람 1명, 지게차 1대",
                "objects_detected": ["worker", "forklift"],
            },
            "thermal_summary": "작업자와 지게차가 매우 가깝게 접근하고 있음",
            "detected_objects": ["worker", "forklift"],
            "use_sliding_window": True,
            "thermal_matrix": [
                [0.10, 0.20, 0.18],
                [0.22, 0.84, 0.88],
                [0.16, 0.86, 0.90],
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["risk_type"] == "충돌주의"
    assert body["detection_strategy"] == "sliding_window_hybrid"
