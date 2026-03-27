from __future__ import annotations

from dataclasses import dataclass

from models import BoundingBox, InferenceDetection, SlidingWindowConfig


@dataclass
class SlidingWindowScanResult:
    detections: list[InferenceDetection]
    scanned_windows: int
    hotspot_count: int
    max_heat_score: float


def _clamp_heat(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalized_bbox(
    *,
    row: int,
    col: int,
    window_size: int,
    rows: int,
    cols: int,
) -> BoundingBox:
    return BoundingBox(
        left=col / cols,
        top=row / rows,
        width=min(window_size / cols, 1.0),
        height=min(window_size / rows, 1.0),
    )


def _valid_matrix(matrix: list[list[float]]) -> bool:
    if not matrix or not matrix[0]:
        return False
    expected = len(matrix[0])
    return all(len(row) == expected for row in matrix)


def scan_thermal_matrix(
    matrix: list[list[float]],
    config: SlidingWindowConfig,
) -> SlidingWindowScanResult:
    if not _valid_matrix(matrix):
        return SlidingWindowScanResult(
            detections=[],
            scanned_windows=0,
            hotspot_count=0,
            max_heat_score=0.0,
        )

    rows = len(matrix)
    cols = len(matrix[0])
    stride = max(1, int(config.stride))
    ordered_candidates: list[tuple[float, int, int, int, int]] = []
    scanned_windows = 0
    max_heat_score = 0.0

    for size in sorted({max(1, int(size)) for size in config.window_sizes}):
        if size > rows or size > cols:
            continue

        for row in range(0, rows - size + 1, stride):
            for col in range(0, cols - size + 1, stride):
                scanned_windows += 1
                values = [
                    _clamp_heat(matrix[row_idx][col_idx])
                    for row_idx in range(row, row + size)
                    for col_idx in range(col, col + size)
                ]
                avg_heat = sum(values) / len(values)
                hot_cells = sum(1 for value in values if value >= config.heat_threshold)
                hotspot_ratio = hot_cells / len(values)
                score = (avg_heat * 0.7) + (hotspot_ratio * 0.3)
                max_heat_score = max(max_heat_score, score)

                if avg_heat >= config.heat_threshold and hot_cells >= config.min_hot_cells:
                    ordered_candidates.append((score, row, col, size, hot_cells))

    ordered_candidates.sort(key=lambda item: item[0], reverse=True)
    limited = ordered_candidates[: max(1, config.max_windows)]

    detections = [
        InferenceDetection(
            label=f"thermal_hotspot_w{size}",
            confidence=round(min(0.99, 0.55 + (score * 0.4)), 4),
            heat_score=round(score, 4),
            bbox=_normalized_bbox(row=row, col=col, window_size=size, rows=rows, cols=cols),
        )
        for score, row, col, size, _hot_cells in limited
    ]

    return SlidingWindowScanResult(
        detections=detections,
        scanned_windows=scanned_windows,
        hotspot_count=len(detections),
        max_heat_score=round(max_heat_score, 4),
    )
