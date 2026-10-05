"""Object detection using OpenCV's DNN module with a MobileNet-SSD (Caffe) model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

LABELS = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
          "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant",
          "sheep", "sofa", "train", "tvmonitor"]

# Friendlier spoken names
SPOKEN = {"diningtable": "table", "pottedplant": "potted plant", "tvmonitor": "screen",
          "aeroplane": "aeroplane", "motorbike": "motorbike"}


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def spoken(self) -> str:
        return SPOKEN.get(self.label, self.label)

    @property
    def center_x(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def area(self) -> int:
        return max(0, self.x2 - self.x1) * max(0, self.y2 - self.y1)


class ObjectDetector:
    def __init__(self, model_dir: str | Path = "models", conf_threshold: float = 0.5):
        model_dir = Path(model_dir)
        proto, weights = model_dir / "deploy.prototxt", model_dir / "mobilenet_iter_73000.caffemodel"
        if not proto.exists() or not weights.exists():
            raise FileNotFoundError(f"Model files not found in {model_dir}. Run scripts/download_models.sh")
        self.net = cv2.dnn.readNetFromCaffe(str(proto), str(weights))
        self.conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        self.net.setInput(blob)
        out = self.net.forward()
        results: list[Detection] = []
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            idx = int(out[0, 0, i, 1])
            if conf < self.conf_threshold or not 0 < idx < len(LABELS):
                continue
            x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            results.append(Detection(LABELS[idx], conf, max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
        return results

    @staticmethod
    def annotate(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
        img = frame.copy()
        for d in detections:
            cv2.rectangle(img, (d.x1, d.y1), (d.x2, d.y2), (0, 200, 0), 2)
            cv2.putText(img, f"{d.spoken} {d.confidence:.0%}", (d.x1, max(15, d.y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        return img
