from __future__ import annotations

from math import hypot
from typing import Any


class SimpleTracker:
    def __init__(self, match_distance: float = 80.0, motion_distance: float = 8.0) -> None:
        self.match_distance = match_distance
        self.motion_distance = motion_distance
        self.next_id = 1
        self.tracks: list[dict[str, Any]] = []

    def update(
        self,
        detections: list[dict[str, Any]],
        frame_index: int,
        timestamp_sec: float,
    ) -> list[dict[str, Any]]:
        matched_track_ids: set[int] = set()
        updated_tracks = []

        for detection in detections:
            track = self.find_matching_track(detection, matched_track_ids)

            if track is None:
                track = self.create_track(detection, frame_index)
                self.tracks.append(track)
            else:
                self.update_track(track, detection, frame_index)

            matched_track_ids.add(track["object_id"])
            updated_tracks.append(track)

        return updated_tracks

    def find_matching_track(
        self,
        detection: dict[str, Any],
        matched_track_ids: set[int],
    ) -> dict[str, Any] | None:
        best_track = None
        best_distance = self.match_distance

        for track in self.tracks:
            if track["object_id"] in matched_track_ids:
                continue
            if track["class"] != detection["class"]:
                continue

            distance = center_distance(track["last_center"], detection["center"])
            if distance < best_distance:
                best_distance = distance
                best_track = track

        return best_track

    def create_track(self, detection: dict[str, Any], frame_index: int) -> dict[str, Any]:
        track = {
            "object_id": self.next_id,
            "class": detection["class"],
            "first_seen_frame": frame_index,
            "last_seen_frame": frame_index,
            "last_bbox": detection["bbox"],
            "last_center": detection["center"],
            "best_confidence": detection["confidence"],
            "motion_points": [{"frame": frame_index, "state": "stationary"}],
            "interactions": [],
        }
        self.next_id += 1
        return track

    def update_track(
        self,
        track: dict[str, Any],
        detection: dict[str, Any],
        frame_index: int,
    ) -> None:
        distance = center_distance(track["last_center"], detection["center"])
        state = "moving" if distance >= self.motion_distance else "stationary"

        track["last_seen_frame"] = frame_index
        track["last_bbox"] = detection["bbox"]
        track["last_center"] = detection["center"]
        track["best_confidence"] = max(track["best_confidence"], detection["confidence"])
        track["motion_points"].append({"frame": frame_index, "state": state})

    def export_tracks(self) -> list[dict[str, Any]]:
        output = []

        for track in self.tracks:
            output.append(
                {
                    "object_id": track["object_id"],
                    "class": track["class"],
                    "first_seen_frame": track["first_seen_frame"],
                    "last_seen_frame": track["last_seen_frame"],
                    "best_confidence": round(track["best_confidence"], 4),
                    "last_bbox": track["last_bbox"],
                    "interacted": len(track["interactions"]) > 0,
                    "motion_history": make_motion_history(track["motion_points"]),
                    "interactions": make_interaction_history(track["interactions"]),
                }
            )

        return output


def center_distance(a: list[float], b: list[float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def make_motion_history(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(points) == 0:
        return []

    history = []
    start_frame = points[0]["frame"]
    current_state = points[0]["state"]
    last_frame = start_frame

    for point in points[1:]:
        if point["state"] != current_state:
            history.append({"frame_range": [start_frame, last_frame], "state": current_state})
            start_frame = point["frame"]
            current_state = point["state"]
        last_frame = point["frame"]

    history.append({"frame_range": [start_frame, last_frame], "state": current_state})
    return history


def make_interaction_history(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(events) == 0:
        return []

    history = []
    start_frame = events[0]["frame_index"]
    end_frame = events[0]["frame_index"]
    person_id = events[0]["interacted_by_person"]
    best_confidence = events[0]["confidence"]
    reason = events[0]["reason"]

    for event in events[1:]:
        same_person = event["interacted_by_person"] == person_id
        close_in_time = event["frame_index"] - end_frame <= 10

        if not same_person or not close_in_time:
            history.append(
                {
                    "interacted_by_person": person_id,
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                    "confidence_max": round(best_confidence, 4),
                    "reason": reason,
                }
            )
            start_frame = event["frame_index"]
            person_id = event["interacted_by_person"]
            best_confidence = event["confidence"]
            reason = event["reason"]

        end_frame = event["frame_index"]
        if event["confidence"] > best_confidence:
            best_confidence = event["confidence"]
            reason = event["reason"]

    history.append(
        {
            "interacted_by_person": person_id,
            "frame_start": start_frame,
            "frame_end": end_frame,
            "confidence_max": round(best_confidence, 4),
            "reason": reason,
        }
    )
    return history
