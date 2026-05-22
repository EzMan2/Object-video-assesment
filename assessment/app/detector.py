from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_MODEL = "yolov8n.pt"
CLASS_OVERRIDES = {
    "laptop": "cord",
}


class Detector:
    def __init__(self, model_name: str = DEFAULT_MODEL, confidence: float = 0.35) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        results = self.model.predict(frame, conf=self.confidence, verbose=False)
        detections: list[dict[str, Any]] = []

        if not results:
            return detections

        result = results[0]
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            class_name = names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].cpu().numpy().astype(float).tolist()
            detections.append(
                {
                    "class": CLASS_OVERRIDES.get(class_name, class_name),
                    "confidence": round(float(box.conf[0].item()), 4),
                    "bbox": [round(value, 2) for value in xyxy],
                    "center": [
                        round((xyxy[0] + xyxy[2]) / 2, 2),
                        round((xyxy[1] + xyxy[3]) / 2, 2),
                    ],
                }
            )

        return detections
