"""Command line entry point.

  python -m assistant.cli image samples/messi5.jpg --save docs/annotated.jpg
  python -m assistant.cli webcam            # press q to quit
"""
from __future__ import annotations

import argparse
import time

import cv2

from .describer import describe
from .detector import ObjectDetector
from .speech import Speaker


def run_image(path: str, args) -> None:
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    det = ObjectDetector(args.models, args.conf)
    t0 = time.perf_counter()
    found = det.detect(frame)
    ms = (time.perf_counter() - t0) * 1000
    h, w = frame.shape[:2]
    print(f"{len(found)} object(s) in {ms:.0f} ms")
    for d in found:
        print(f"  {d.label:12s} {d.confidence:.0%}  box=({d.x1},{d.y1},{d.x2},{d.y2})")
    message = describe(found, w, h)
    print("Guidance:", message)
    if args.save:
        cv2.imwrite(args.save, det.annotate(frame, found))
        print("Saved", args.save)


def run_webcam(args) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera")
    det, speaker = ObjectDetector(args.models, args.conf), Speaker(cooldown=args.cooldown)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            found = det.detect(frame)
            h, w = frame.shape[:2]
            if found:
                speaker.say(describe(found, w, h))
            cv2.imshow("Smart Shopping Assistant (q to quit)", det.annotate(frame, found))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    p = argparse.ArgumentParser(description="Smart Shopping Assistant for the Visually Impaired")
    p.add_argument("--models", default="models")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    sub = p.add_subparsers(dest="mode", required=True)
    pi = sub.add_parser("image"); pi.add_argument("path"); pi.add_argument("--save")
    pw = sub.add_parser("webcam"); pw.add_argument("--camera", type=int, default=0)
    pw.add_argument("--cooldown", type=float, default=4.0)
    args = p.parse_args()
    run_image(args.path, args) if args.mode == "image" else run_webcam(args)


if __name__ == "__main__":
    main()
