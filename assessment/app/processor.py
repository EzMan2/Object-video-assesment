from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2

from app.detector import Detector
from app.interaction import InteractionDetector
from app.storage import KEYFRAME_DIR, get_task, save_result, update_task
from app.tracker import SimpleTracker


def process_video(
    task_id: str,
    video_path: str | Path,
    sample_every_n_frames: int = 5,
    save_keyframes: bool = True,
) -> dict[str, Any]:
    video_path = Path(video_path)
    update_task(task_id, status="processing", progress=0.0)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = frame_count / fps if fps else 0.0

    detector = Detector()
    tracker = SimpleTracker()
    interaction_detector = InteractionDetector()
    keyframe_dir = KEYFRAME_DIR / task_id
    if save_keyframes:
        keyframe_dir.mkdir(parents=True, exist_ok=True)

    processed_frames = 0
    keyframes_by_object: dict[int, dict[str, Any]] = {}
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames != 0:
                frame_index += 1
                continue

            timestamp_sec = frame_index / fps if fps else 0.0
            detections = detector.detect(frame)
            people = [detection for detection in detections if detection["class"] == "person"]
            non_people = [detection for detection in detections if detection["class"] != "person"]

            tracked_objects = tracker.update(non_people, frame_index, timestamp_sec)
            for track in tracked_objects:
                interaction = interaction_detector.update(
                    object_id=track["object_id"],
                    object_bbox=track["last_bbox"],
                    person_detections=people,
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                )
                if interaction:
                    track["interactions"].append(interaction)
                    previous = keyframes_by_object.get(track["object_id"])
                    if save_keyframes and (
                        previous is None or interaction["confidence"] > previous["confidence"]
                    ):
                        keyframe_path = keyframe_dir / f"object_{track['object_id']}_best_interaction.jpg"
                        cv2.imwrite(str(keyframe_path), frame)
                        keyframes_by_object[track["object_id"]] = {
                            "object_id": track["object_id"],
                            "frame_index": frame_index,
                            "confidence": interaction["confidence"],
                            "path": str(keyframe_path),
                        }

            processed_frames += 1
            if processed_frames % 10 == 0 and frame_count:
                update_task(task_id, progress=round(frame_index / frame_count, 3))

            frame_index += 1
    finally:
        capture.release()

    task = get_task(task_id) or {}
    result = {
        "task_id": task_id,
        "videoMetadata": {
            "filename": task.get("filename", video_path.name),
            "fps": round(fps, 3),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": round(duration_sec, 3),
            "sample_every_n_frames": sample_every_n_frames,
            "processed_frames": processed_frames,
        },
        "objectsDetected": tracker.export_tracks(),
        "keyframes": list(keyframes_by_object.values()),
    }
    result_path = save_result(task_id, result)
    update_task(
        task_id,
        status="completed",
        progress=1.0,
        result_path=str(result_path),
        object_count=len(result["objectsDetected"]),
    )
    return result


def process_video_task(task_id: str, video_path: str) -> None:
    try:
        process_video(task_id, video_path)
    except Exception as exc:
        update_task(task_id, status="failed", error=str(exc))
