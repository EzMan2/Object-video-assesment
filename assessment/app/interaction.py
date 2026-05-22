from __future__ import annotations

from math import hypot
from typing import Any


def bbox_iou(a: list[float], b: list[float]) -> float:
    x_left = max(a[0], b[0])
    y_top = max(a[1], b[1])
    x_right = min(a[2], b[2])
    y_bottom = min(a[3], b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def bbox_distance(a: list[float], b: list[float]) -> float:
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2
    return hypot(ax - bx, ay - by)


class InteractionDetector:
    def __init__(
        self,
        near_distance_px: float = 90.0,
        overlap_threshold: float = 0.02,
        consecutive_frames: int = 2,
    ) -> None:
        self.near_distance_px = near_distance_px
        self.overlap_threshold = overlap_threshold
        self.consecutive_frames = consecutive_frames
        self._near_counts: dict[int, int] = {}

    def update(
        self,
        object_id: int,
        object_bbox: list[float],
        person_detections: list[dict[str, Any]],
        frame_index: int,
        timestamp_sec: float,
    ) -> dict[str, Any] | None:
        best_confidence = 0.0
        best_person_id = 0
        best_reason = ""

        for person_id, person in enumerate(person_detections):
            person_bbox = person["bbox"]
            overlap = bbox_iou(object_bbox, person_bbox)
            distance = bbox_distance(object_bbox, person_bbox)

            if overlap >= self.overlap_threshold:
                confidence = min(1.0, 0.65 + overlap)
                reason = "overlap_with_person"
            elif distance <= self.near_distance_px:
                confidence = max(0.1, 1.0 - (distance / self.near_distance_px)) * 0.65
                reason = "near_person"
            else:
                confidence = 0.0
                reason = ""

            if confidence > best_confidence:
                best_confidence = confidence
                best_person_id = person_id
                best_reason = reason

        if best_confidence <= 0:
            self._near_counts[object_id] = 0
            return None

        self._near_counts[object_id] = self._near_counts.get(object_id, 0) + 1
        if self._near_counts[object_id] < self.consecutive_frames:
            return None

        return {
            "frame_index": frame_index,
            "timestamp_sec": round(timestamp_sec, 3),
            "interacted_by_person": best_person_id,
            "confidence": round(best_confidence, 4),
            "reason": best_reason,
            "consecutive_frames": self._near_counts[object_id],
        }
