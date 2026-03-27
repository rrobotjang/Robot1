from __future__ import annotations

from models import SlidingWindowConfig
from sliding_window import scan_thermal_matrix


def test_scan_thermal_matrix_detects_hotspot_windows() -> None:
    matrix = [
        [0.10, 0.12, 0.14, 0.15],
        [0.18, 0.84, 0.90, 0.24],
        [0.20, 0.86, 0.92, 0.22],
        [0.14, 0.16, 0.20, 0.18],
    ]
    result = scan_thermal_matrix(
        matrix,
        SlidingWindowConfig(window_sizes=[2, 3], stride=1, heat_threshold=0.72, min_hot_cells=2),
    )

    assert result.scanned_windows > 0
    assert result.hotspot_count >= 1
    assert result.max_heat_score >= 0.72
    assert result.detections[0].bbox is not None
