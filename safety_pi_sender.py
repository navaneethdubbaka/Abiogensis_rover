"""
Raspberry Pi: camera -> JPEG -> PC safety monitor /safety/ingest.

On the Pi, install: pip install -r requirements_safety_pi.txt
Run:  python safety_pi_sender.py
Or:   python safety_pi_sender.py --server http://PC_LAN_IP:8766

PC must be running: python -m safety_monitor.main
"""

from __future__ import annotations

import argparse
import os
import time

import cv2
import httpx

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi camera -> PC safety ingest")
    parser.add_argument(
        "--server",
        default=os.getenv("SAFETY_PC_BASE_URL", "http://127.0.0.1:8766").rstrip("/"),
        help="PC safety server base URL (no trailing slash)",
    )
    parser.add_argument(
        "--camera-id",
        default=os.getenv("SAFETY_CAMERA_ID", "pi_cam"),
        help="Logical camera id for multi-camera on PC",
    )
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--width", type=int, default=int(os.getenv("FRAME_W", "640")))
    parser.add_argument("--height", type=int, default=int(os.getenv("FRAME_H", "480")))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.getenv("JPEG_QUALITY", "80")))
    parser.add_argument(
        "--interval",
        type=float,
        default=float(os.getenv("SAFETY_UPLOAD_INTERVAL_SEC", "7.5")),
        help="Seconds between uploads",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("SAFETY_HTTP_TIMEOUT", "15.0")),
    )
    parser.add_argument(
        "--token",
        default=os.getenv("SAFETY_INGEST_TOKEN", ""),
        help="If PC has SAFETY_INGEST_SECRET, pass the same value (sent as X-Safety-Token)",
    )
    args = parser.parse_args()

    url = f"{args.server}/safety/ingest"
    headers = {}
    if args.token.strip():
        headers["X-Safety-Token"] = args.token.strip()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv("CAMERA_BUFFER_SIZE", "1")))

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.jpeg_quality))]
    interval = max(0.5, float(args.interval))

    grab_flush = max(0, int(os.getenv("CAMERA_GRAB_FLUSH", "1")))

    try:
        with httpx.Client(timeout=args.timeout) as client:
            while True:
                t0 = time.monotonic()
                for _ in range(grab_flush):
                    cap.grab()
                ok, frame = cap.read()
                if not ok:
                    time.sleep(1.0)
                    continue
                ok_buf, buf = cv2.imencode(".jpg", frame, encode_params)
                if not ok_buf:
                    time.sleep(0.2)
                    continue
                jpeg = buf.tobytes()
                try:
                    r = client.post(
                        url,
                        files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                        data={"camera_id": args.camera_id},
                        headers=headers,
                    )
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"[safety_pi_sender] upload failed: {e}")
                    time.sleep(min(5.0, interval))
                elapsed = time.monotonic() - t0
                sleep_for = max(0.0, interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        cap.release()


if __name__ == "__main__":
    main()
