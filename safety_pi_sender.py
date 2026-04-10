"""
Raspberry Pi OS: camera -> JPEG -> PC safety monitor /safety/ingest.

USB camera: same style as sonic_lang/robot_listener.py — OpenCV VideoCapture(index) and,
for live view, an MJPEG page in your browser (no Qt/imshow; avoids QStandardPaths/GUI issues).

On the Pi: pip install -r requirements_safety_pi.txt

Preview: open http://<pi-ip>:9080/ (or http://127.0.0.1:9080/ on the Pi) in Chromium.
Use --no-preview or HEADLESS=1 to disable the local preview server.

CSI ribbon camera: --picamera2 (not USB).

PC must be running: python -m safety_monitor.main
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from types import SimpleNamespace

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

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


def start_mjpeg_preview_thread(
    cap: cv2.VideoCapture,
    lock: threading.Lock,
    *,
    port: int,
    stream_jpeg_quality: int,
) -> None:
    """Same pattern as robot_listener.py: /camera/stream MJPEG + simple HTML page."""
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, StreamingResponse
        import uvicorn
    except ImportError:
        print(
            "[safety_pi_sender] Install preview deps: pip install fastapi uvicorn",
            file=sys.stderr,
        )
        return

    app = FastAPI()
    enc = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, stream_jpeg_quality))]

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Safety Pi camera</title>
<style>body{{margin:0;background:#111;color:#ccc;font-family:sans-serif;}}
header{{padding:8px 12px;background:#222;}} img{{width:100%;height:auto;display:block;}}</style>
</head><body>
<header>Safety monitor — live feed (robot_listener-style MJPEG)</header>
<img src="/camera/stream" alt="camera">
</body></html>"""

    def generate_mjpeg():
        while True:
            with lock:
                ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            ok2, buf = cv2.imencode(".jpg", frame, enc)
            if not ok2:
                time.sleep(0.05)
                continue
            blob = buf.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + blob + b"\r\n"
            time.sleep(0.033)

    @app.get("/camera/stream")
    async def camera_stream():
        return StreamingResponse(
            generate_mjpeg(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    cfg = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(cfg)

    def run():
        try:
            server.run()
        except Exception as e:
            print(f"[safety_pi_sender] preview server stopped: {e}", file=sys.stderr)

    threading.Thread(target=run, daemon=True).start()
    time.sleep(0.4)
    base = f"http://127.0.0.1:{port}/"
    print(f"[safety_pi_sender] Live camera (browser): {base}", flush=True)
    print(f"[safety_pi_sender] From another device: http://<this-pi-ip>:{port}/", flush=True)
    if _env_bool("SAFETY_PI_OPEN_BROWSER", True) and os.environ.get("DISPLAY"):
        try:
            import webbrowser

            webbrowser.open(base)
        except Exception:
            pass


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
        for preview_kind, label in ((Preview.DRM, "DRM"), (Preview.QTGL, "Qt GL")):
            try:
                picam2.start_preview(preview_kind)
                print(f"[safety_pi_sender] Picamera2 preview: {label}", flush=True)
                break
            except Exception:
                continue

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
                try:
                    r = client.post(
                        url,
                        files={"image": ("frame.jpg", jpeg, "image/jpeg")},
                        data={"camera_id": args.camera_id},
                        headers=headers,
                    )
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"[safety_pi_sender] upload failed: {e}", flush=True)
                    time.sleep(min(5.0, interval))
                elapsed = time.monotonic() - t0
                sleep_for = max(0.0, interval - elapsed)
                if sleep_for > 0:
                    time.sleep(sleep_for)
    finally:
        for fn in (picam2.stop_preview, picam2.stop, picam2.close):
            try:
                fn()
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
    # Same as robot_listener CameraController: plain index open
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv("CAMERA_BUFFER_SIZE", "1")))
    if not cap.isOpened():
        print("[safety_pi_sender] Failed to open camera", file=sys.stderr)
        return

    lock = threading.Lock()
    preview_port = int(os.getenv("SAFETY_PI_PREVIEW_PORT", "9080"))
    stream_q = int(os.getenv("SAFETY_PI_STREAM_JPEG_QUALITY", str(args.jpeg_quality)))

    if preview_wanted:
        start_mjpeg_preview_thread(cap, lock, port=preview_port, stream_jpeg_quality=stream_q)

    try:
        with httpx.Client(timeout=args.timeout) as client:
            while True:
                t0 = time.monotonic()
                with lock:
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
        help="Do not start local MJPEG preview server",
    )
    parser.add_argument("--opencv", action="store_true", help="Force OpenCV USB path")
    parser.add_argument(
        "--picamera2",
        action="store_true",
        help="CSI ribbon camera (Picamera2), not USB",
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

    use_picamera2 = args.picamera2 or _env_bool("SAFETY_PI_USE_PICAMERA2", False)
    if sys.platform == "linux" and use_picamera2 and not args.opencv:
        try:
            run_picamera2(args, url, headers, preview_wanted, encode_params, interval, grab_flush)
            return
        except ImportError:
            print("[safety_pi_sender] pip install picamera2", file=sys.stderr)
        except Exception as e:
            print(f"[safety_pi_sender] Picamera2 failed ({e}); using OpenCV.", file=sys.stderr)

    run_opencv(args, url, headers, preview_wanted, encode_params, interval, grab_flush)


if __name__ == "__main__":
    main()
