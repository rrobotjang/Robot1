from __future__ import annotations

import re
from typing import Iterable

from models import (
    BoundingBox,
    CVInferenceRequest,
    CVInferenceResponse,
    InferenceDetection,
    RiskLevel,
    SiteProfile,
)
from sliding_window import scan_thermal_matrix


ALIASES = {
    "worker": ["worker", "사람", "작업자", "인원"],
    "forklift": ["forklift", "지게차"],
    "pallet": ["pallet", "적재물", "파렛트", "화물"],
    "pedestrian": ["pedestrian", "보행자"],
    "car": ["car", "차량", "승용차"],
    "barrier": ["barrier", "콘", "장애물", "공사장애물"],
    "truck": ["truck", "화물차", "트럭"],
    "bicycle": ["bicycle", "자전거"],
    "stopped_vehicle": ["정차", "stopped", "정차차량", "비정상정차"],
}


DETECTION_LIBRARY = {
    "worker": (0.97, 0.86, BoundingBox(left=0.14, top=0.18, width=0.22, height=0.48)),
    "forklift": (0.96, 0.91, BoundingBox(left=0.52, top=0.22, width=0.31, height=0.45)),
    "pallet": (0.91, 0.74, BoundingBox(left=0.64, top=0.62, width=0.2, height=0.14)),
    "pedestrian": (0.95, 0.84, BoundingBox(left=0.28, top=0.2, width=0.18, height=0.46)),
    "car": (0.94, 0.8, BoundingBox(left=0.58, top=0.42, width=0.28, height=0.24)),
    "barrier": (0.93, 0.72, BoundingBox(left=0.46, top=0.48, width=0.22, height=0.18)),
    "truck": (0.9, 0.79, BoundingBox(left=0.56, top=0.36, width=0.3, height=0.28)),
    "bicycle": (0.88, 0.68, BoundingBox(left=0.18, top=0.44, width=0.16, height=0.2)),
    "stopped_vehicle": (0.87, 0.7, BoundingBox(left=0.54, top=0.4, width=0.29, height=0.24)),
}


def _contains_any(text: str, keywords: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _collect_objects(payload: CVInferenceRequest) -> list[str]:
    corpus = " ".join(
        [
            payload.camera.stream_hint,
            payload.thermal_summary,
            payload.operator_note or "",
            " ".join(payload.camera.objects_detected),
            " ".join(payload.detected_objects),
        ]
    ).lower()

    collected: list[str] = []
    for canonical, aliases in ALIASES.items():
        if canonical in payload.detected_objects:
            collected.append(canonical)
            continue
        if _contains_any(corpus, aliases):
            collected.append(canonical)

    if not collected:
        collected.extend(payload.camera.objects_detected)

    return list(dict.fromkeys(collected))


def _build_semantic_detections(objects: list[str]) -> list[InferenceDetection]:
    detections = []
    for label in objects:
        confidence, heat_score, bbox = DETECTION_LIBRARY.get(
            label,
            (0.78, 0.55, BoundingBox(left=0.35, top=0.35, width=0.22, height=0.22)),
        )
        detections.append(
            InferenceDetection(
                label=label,
                confidence=confidence,
                heat_score=heat_score,
                bbox=bbox,
            )
        )
    return detections


def _evaluate(
    payload: CVInferenceRequest,
    detected_objects: list[str],
    hotspot_count: int,
    hotspot_heat: float,
) -> tuple[str, RiskLevel, str, str, float | None]:
    text = payload.thermal_summary.lower()
    objects = set(detected_objects)
    zone = payload.camera.zone

    if payload.profile == SiteProfile.LOGISTICS:
        if {"worker", "forklift"}.issubset(objects):
            level = RiskLevel.CRITICAL if _contains_any(text, ["근접", "접촉", "위험"]) else RiskLevel.HIGH
            return (
                "충돌주의",
                level,
                f"{zone}에서 작업자와 지게차의 이동 경로가 겹쳐 즉시 대응이 필요합니다.",
                "현장 방송과 지게차 감속 모드를 즉시 적용하고 작업자를 안전 구역으로 유도하세요.",
                1.8,
            )

        if zone == "출입제한구역" and "worker" in objects:
            return (
                "구역침범",
                RiskLevel.HIGH,
                "출입 제한구역에서 작업자 접근이 감지되어 이동 동선 재안내가 필요합니다.",
                "출입 제한구역 경광등을 활성화하고 현장 담당자를 호출하세요.",
                None,
            )

        if {"pallet", "forklift"}.issubset(objects):
            return (
                "적재충돌주의",
                RiskLevel.MEDIUM,
                f"{zone}에서 적재물과 차량 동선이 겹쳐 작업 혼선이 발생하고 있습니다.",
                "임시 적재물 위치를 조정하고 지게차 통행 구간을 다시 분리하세요.",
                3.4,
            )

        if hotspot_count >= 2 and hotspot_heat >= 0.78:
            return (
                "열점집중경고",
                RiskLevel.HIGH,
                f"{zone}에서 다중 열점 구역이 스캔되어 현장 확인이 필요합니다.",
                "근접 작업을 일시 중지하고 적재물, 모터, 작업자 밀집 여부를 현장에서 점검하세요.",
                None,
            )

        if hotspot_count >= 1 and hotspot_heat >= 0.72:
            return (
                "국부과열주의",
                RiskLevel.MEDIUM,
                f"{zone}에서 국부 고온 영역이 탐지되어 장비 및 작업 상태 확인이 필요합니다.",
                "고온 구역 반경을 통제하고 센서 오염 여부와 장비 발열 상태를 확인하세요.",
                None,
            )

        return (
            "이상상황",
            RiskLevel.MEDIUM,
            f"{zone}에서 평소와 다른 열 패턴이 감지되어 현장 확인이 필요합니다.",
            "현장 관리자에게 확인 요청을 보내고 카메라 대비 설정을 점검하세요.",
            None,
        )

    if {"pedestrian", "car"}.issubset(objects):
        level = RiskLevel.CRITICAL if _contains_any(text, ["근접", "횡단", "급접근"]) else RiskLevel.HIGH
        return (
            "보행자경고",
            level,
            f"{zone}에서 보행자와 차량 이동 벡터가 교차하고 있습니다.",
            "점멸 경고등과 가변 표지판을 활성화하고 접근 차량 속도를 즉시 낮추세요.",
            2.6,
        )

    if {"barrier"}.issubset(objects) or {"truck", "barrier"}.issubset(objects):
        return (
            "돌발장애물",
            RiskLevel.HIGH,
            f"{zone} 주행 차선에서 장애물 또는 공사 장비가 탐지되었습니다.",
            "차선 통제 알림을 송출하고 현장 관리 차량을 출동시키세요.",
            None,
        )

    if "stopped_vehicle" in objects:
        return (
            "이상정차",
            RiskLevel.MEDIUM,
            f"{zone}에서 비정상 정차 패턴이 감지되었습니다.",
            "정차 구역 방송 안내를 송출하고 관제 담당자 확인을 요청하세요.",
            None,
        )

    if hotspot_count >= 2 and hotspot_heat >= 0.78:
        return (
            "열점이상경고",
            RiskLevel.HIGH,
            f"{zone}에서 복수의 열점 구역이 연속 스캔되어 돌발 상황 가능성이 있습니다.",
            "저속 주행 경고를 송출하고 현장 카메라와 순찰 차량으로 즉시 재확인하세요.",
            None,
        )

    if hotspot_count >= 1 and hotspot_heat >= 0.72:
        return (
            "국부열점주의",
            RiskLevel.MEDIUM,
            f"{zone}에서 국부 열점이 탐지되어 저시야 구간의 이상 객체 확인이 필요합니다.",
            "가변 표지판 경고를 활성화하고 다음 프레임에서 동일 위치 재감지를 확인하세요.",
            None,
        )

    return (
        "저시야경고",
        RiskLevel.MEDIUM,
        f"{zone}에서 저조도 환경 이상 패턴이 탐지되었습니다.",
        "카메라 보정 상태를 점검하고 현장 순찰을 요청하세요.",
        None,
    )


def run_cv_inference(payload: CVInferenceRequest) -> CVInferenceResponse:
    objects = _collect_objects(payload)
    semantic_detections = _build_semantic_detections(objects)
    window_scan = (
        scan_thermal_matrix(payload.thermal_matrix, payload.sliding_window)
        if payload.use_sliding_window
        else None
    )
    window_detections = window_scan.detections if window_scan else []
    risk_type, risk_level, summary, recommended_action, distance_m = _evaluate(
        payload,
        objects,
        window_scan.hotspot_count if window_scan else 0,
        window_scan.max_heat_score if window_scan else 0.0,
    )
    if payload.use_sliding_window and window_scan and window_scan.hotspot_count > 0:
        strategy = "sliding_window_hybrid"
    elif payload.use_sliding_window:
        strategy = "sliding_window"
    else:
        strategy = "semantic"

    return CVInferenceResponse(
        detections=semantic_detections + window_detections,
        risk_level=risk_level,
        risk_type=risk_type,
        summary=summary,
        recommended_action=recommended_action,
        distance_m=distance_m,
        object_labels=objects,
        detection_strategy=strategy,
        scanned_windows=window_scan.scanned_windows if window_scan else 0,
        hotspot_count=window_scan.hotspot_count if window_scan else 0,
    )
