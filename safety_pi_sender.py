"""
Raspberry Pi: camera -> JPEG -> PC safety monitor /safety/ingest.

On the Pi, install: pip install -r requirements_safety_pi.txt
Run:  python safety_pi_sender.py
Or:   python safety_pi_sender.py --server http://PC_LAN_IP:8766

PC must be running: python -m safety_monitor.main

Shows a live camera window by default (Esc to exit). Use --no-preview or HEADLESS=1
without a display. Requires opencv-python (GUI); use opencv-python-headless + --no-preview on headless Pi.
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


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


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
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not open a live window (use on SSH/headless Pi)",
    )
    args = parser.parse_args()

    preview = not args.no_preview and not _env_bool("HEADLESS", False)

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
    win = "Safety Pi camera"
    if preview:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def draw_overlay(display, status: str) -> None:
        cv2.putText(
            display,
            f"{args.camera_id} | {status}",
            (8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            display,
            "Esc quit",
            (8, display.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

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
                upload_status = "upload ok"
                try:
                    r = client.post(
                        url,
                        files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                        data={"camera_id": args.camera_id},
                        headers=headers,
                    )
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    upload_status = f"net err"
                    print(f"[safety_pi_sender] upload failed: {e}")
                    time.sleep(min(5.0, interval))
                if preview:
                    vis = frame.copy()
                    draw_overlay(vis, upload_status)
                    cv2.imshow(win, vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break
                elapsed = time.monotonic() - t0
                sleep_for = max(0.0, interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
