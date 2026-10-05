"""Turns raw detections into short spoken guidance. Pure Python, no OpenCV, easy to unit-test."""
from __future__ import annotations

from .detector import Detection

NEAR_RATIO = 0.25    # box covers >= 25% of the frame  -> "very close"
MEDIUM_RATIO = 0.06  # box covers >= 6% of the frame   -> "nearby", else "further away"


def position(det: Detection, frame_width: int) -> str:
    """Left / ahead / right based on which third of the frame the box centre falls in."""
    third = frame_width / 3
    if det.center_x < third:
        return "on your left"
    if det.center_x > 2 * third:
        return "on your right"
    return "ahead"


def proximity(det: Detection, frame_width: int, frame_height: int) -> str:
    """Rough distance bucket from how much of the frame the object fills (a heuristic, not a measurement)."""
    ratio = det.area / float(frame_width * frame_height)
    if ratio >= NEAR_RATIO:
        return "very close"
    if ratio >= MEDIUM_RATIO:
        return "nearby"
    return "further away"


def describe(detections: list[Detection], frame_width: int, frame_height: int, max_items: int = 3) -> str:
    """Return a sentence describing the most relevant detections (largest/closest first)."""
    if not detections:
        return "No objects detected."
    ranked = sorted(detections, key=lambda d: d.area, reverse=True)[:max_items]
    parts = [f"{d.spoken} {position(d, frame_width)}, {proximity(d, frame_width, frame_height)}" for d in ranked]
    return "; ".join(parts) + "."
