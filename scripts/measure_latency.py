"""Measures detector latency on one image. Usage: python scripts/measure_latency.py samples/messi5.jpg [runs]"""
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from assistant.detector import ObjectDetector  # noqa: E402

path = sys.argv[1]
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
frame = cv2.imread(path)
det = ObjectDetector()
for _ in range(5):  # warm-up
    det.detect(frame)
times = []
for _ in range(runs):
    t = time.perf_counter()
    det.detect(frame)
    times.append((time.perf_counter() - t) * 1000)
times.sort()
print(f"image={path} size={frame.shape[1]}x{frame.shape[0]} runs={runs}")
print(f"median={statistics.median(times):.1f} ms  p95={times[int(0.95 * runs) - 1]:.1f} ms  "
      f"-> about {1000 / statistics.median(times):.1f} FPS (detector only, CPU)")
