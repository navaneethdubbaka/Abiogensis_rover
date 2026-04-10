"""
Raspberry Pi OS: camera -> JPEG -> PC safety monitor /safety/ingest.

Tested against **Raspberry Pi OS** (Desktop/Lite). Default path: **USB webcam** via OpenCV
(`VideoCapture` + V4L2 on Pi OS, optional `imshow` preview).

On the Pi: pip install -r requirements_safety_pi.txt

Official **CSI / ribbon** camera only (not USB): pip install picamera2, then --picamera2 or SAFETY_PI_USE_PICAMERA2=1.

Run:  python safety_pi_sender.py --server http://PC_LAN_IP:8766

PC must be running: python -m safety_monitor.main

Live preview on Pi OS: `sudo apt install python3-opencv` + venv `--system-site-packages` (see safety_monitor/README.md).
Pip `opencv-python-headless` cannot show a window.

Use --no-preview or HEADLESS=1 to disable the preview window.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from types import SimpleNamespace

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Before import cv2: prefer V4L2 over GStreamer for USB cameras on Raspberry Pi OS.
if sys.platform == "linux":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_V4L2", "999999")
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import httpx

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except AttributeError:
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _open_usb_capture_v4l2(camera_index: int):
    """
    Open USB webcam via V4L2 device path first (avoids GStreamer backend on Raspberry Pi OS).
    Set VIDEO_DEVICE=/dev/videoN to force a path. Falls back to numeric index, then CAP_ANY.
    """
    candidates: list = []
    dev = os.getenv("VIDEO_DEVICE", "").strip()
    if dev:
        candidates.append(dev)
    candidates.append(f"/dev/video{camera_index}")
    candidates.append(camera_index)

    for src in candidates:
        c = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if c.isOpened():
            return c
        c.release()

    print(
        "[safety_pi_sender] V4L2 open failed; trying default backend (may show GStreamer warnings).",
        file=sys.stderr,
    )
    return cv2.VideoCapture(camera_index, cv2.CAP_ANY)


def draw_overlay(display, camera_id: str, status: str) -> None:
    cv2.putText(
        display,
        f"{camera_id} | {status}",
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


def _bgr_from_picamera2_array(arr):
    if arr is None or arr.size == 0:
        raise ValueError("empty frame")
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    c = arr.shape[2]
    if c == 4:
        return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    if c == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def run_picamera2(
    args: SimpleNamespace,
    url: str,
    headers: dict,
    preview_wanted: bool,
    encode_params: list,
    interval: float,
    grab_flush: int,
) -> None:
    from picamera2 import Picamera2, Preview

    picam2 = Picamera2()
    w, h = max(64, args.width), max(64, args.height)
    cfg = picam2.create_preview_configuration(main={"size": (w, h), "format": "RGB888"})
    picam2.configure(cfg)

    if preview_wanted:
        started = False
        candidates = [(Preview.DRM, "DRM"), (Preview.QTGL, "Qt GL")]
        if hasattr(Preview, "QT"):
            candidates.append((Preview.QT, "Qt"))
        for preview_kind, label in candidates:
            try:
                picam2.start_preview(preview_kind)
                print(f"[safety_pi_sender] Camera preview: {label} (Ctrl+C to stop)", flush=True)
                started = True
                break
            except Exception as e:
                print(f"[safety_pi_sender] Preview {label} unavailable: {e}", file=sys.stderr, flush=True)
        if not started:
            print(
                "[safety_pi_sender] No Picamera2 preview backend worked; capture continues, no on-screen feed.",
                file=sys.stderr,
                flush=True)

    picam2.start()

    try:
        with httpx.Client(timeout=args.timeout) as client:
            while True:
                t0 = time.monotonic()
                for _ in range(max(1, grab_flush)):
                    picam2.capture_array("main")
                arr = picam2.capture_array("main")
                bgr = _bgr_from_picamera2_array(arr)
                ok_buf, buf = cv2.imencode(".jpg", bgr, encode_params)
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
                    upload_status = "net err"
                    print(f"[safety_pi_sender] upload failed: {e}", flush=True)
                    time.sleep(min(5.0, interval))
                if preview_wanted:
                    print(f"[safety_pi_sender] {args.camera_id} | {upload_status}", flush=True)
                elapsed = time.monotonic() - t0
                sleep_for = max(0.0, interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        try:
            picam2.stop_preview()
        except Exception:
            pass
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass


def run_opencv(
    args: SimpleNamespace,
    url: str,
    headers: dict,
    preview_wanted: bool,
    encode_params: list,
    interval: float,
    grab_flush: int,
) -> None:
    preview = False
    win = "Safety Pi camera"
    if preview_wanted:
        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            preview = True
        except cv2.error:
            print(
                "[safety_pi_sender] OpenCV has no GUI (common with pip wheels). "
                "For Pi Camera Module use Picamera2: pip install picamera2 and --picamera2. "
                "For USB cam + preview: sudo apt install python3-opencv and venv --system-site-packages.",
                file=sys.stderr,
            )

    backend = cv2.CAP_ANY
    if sys.platform == "linux":
        b = os.getenv("SAFETY_PI_CAP_BACKEND", "v4l2").strip().lower()
        if b in ("", "v4l2", "usb"):
            backend = cv2.CAP_V4L2
        elif b == "any":
            backend = cv2.CAP_ANY

    if sys.platform == "linux" and backend == cv2.CAP_V4L2:
        cap = _open_usb_capture_v4l2(args.camera)
    else:
        cap = cv2.VideoCapture(args.camera, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv("CAMERA_BUFFER_SIZE", "1")))

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
                    upload_status = "net err"
                    print(f"[safety_pi_sender] upload failed: {e}")
                    time.sleep(min(5.0, interval))
                if preview:
                    try:
                        vis = frame.copy()
                        draw_overlay(vis, args.camera_id, upload_status)
                        cv2.imshow(win, vis)
                        key = cv2.waitKey(1) & 0xFF
                        if key == 27:
                            break
                    except cv2.error:
                        print("[safety_pi_sender] imshow failed; disabling preview.", file=sys.stderr)
                        preview = False
                elapsed = time.monotonic() - t0
                sleep_for = max(0.0, interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()


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
        help="Do not start Picamera2 preview or OpenCV window",
    )
    parser.add_argument(
        "--opencv",
        action="store_true",
        help="Force OpenCV path (default for USB cameras; same as not using --picamera2)",
    )
    parser.add_argument(
        "--picamera2",
        action="store_true",
        help="Use Picamera2 (official CSI ribbon camera only, not USB)",
    )
    args = parser.parse_args()

    preview_wanted = not args.no_preview and not _env_bool("HEADLESS", False)

    url = f"{args.server}/safety/ingest"
    headers = {}
    if args.token.strip():
        headers["X-Safety-Token"] = args.token.strip()

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.jpeg_quality))]
    interval = max(0.5, float(args.interval))
    grab_flush = max(0, int(os.getenv("CAMERA_GRAB_FLUSH", "1")))

    use_picamera2 = args.picamera2 or _env_bool(
        "SAFETY_PI_USE_PICAMERA2",
        default=False,
    )
    try_picamera2 = (
        sys.platform == "linux"
        and use_picamera2
        and not args.opencv
    )
    if try_picamera2:
        try:
            run_picamera2(args, url, headers, preview_wanted, encode_params, interval, grab_flush)
            return
        except ImportError:
            print(
                "[safety_pi_sender] picamera2 not installed. Run: pip install picamera2",
                file=sys.stderr,
            )
        except Exception as e:
            print(f"[safety_pi_sender] Picamera2 failed ({e}); falling back to OpenCV.", file=sys.stderr)

    run_opencv(args, url, headers, preview_wanted, encode_params, interval, grab_flush)


if __name__ == "__main__":
    main()
