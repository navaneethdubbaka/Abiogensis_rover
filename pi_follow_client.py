"""
Raspberry Pi thin client: camera -> JPEG -> PC /follow/step -> serial command to Arduino.

Run:  python pi_follow_client.py --server http://PC_LAN_IP:8765

Requires: opencv-python, pyserial, requests
PC must be running: uvicorn pc_follow_server:app --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import cv2
import requests
import serial

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_SERIAL_BY_ID = (
    "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"
)


def resolve_robot_serial_port() -> str:
    for key in ("ROBOT_SERIAL_PORT", "SERIAL_PORT"):
        v = os.getenv(key)
        if v and v.strip():
            return v.strip()
    if os.path.exists(_DEFAULT_SERIAL_BY_ID):
        return _DEFAULT_SERIAL_BY_ID
    for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return _DEFAULT_SERIAL_BY_ID


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pi camera -> PC GPU follow -> Arduino")
    parser.add_argument(
        "--server",
        default=os.getenv("ROBOT_FOLLOW_SERVER", "http://127.0.0.1:8765").rstrip("/"),
        help="PC base URL (no trailing slash)",
    )
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--width", type=int, default=int(os.getenv("FRAME_W", "640")))
    parser.add_argument("--height", type=int, default=int(os.getenv("FRAME_H", "480")))
    parser.add_argument("--jpeg-quality", type=int, default=int(os.getenv("JPEG_QUALITY", "80")))
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("FOLLOW_HTTP_TIMEOUT", "3.0")),
        help="POST read timeout (seconds)",
    )
    parser.add_argument(
        "--post-interval",
        type=float,
        default=float(os.getenv("PI_FOLLOW_POST_INTERVAL_SEC", "0")),
        help="Seconds to sleep after each server round-trip (0 = every frame; use 2–5 for slow VLM)",
    )
    parser.add_argument("--no-preview", action="store_true", help="Do not open cv2 window")
    args = parser.parse_args()

    serial_port = resolve_robot_serial_port()
    baud = int(os.getenv("ROBOT_BAUD_RATE", "115200"))
    arduino = serial.Serial(serial_port, baud, timeout=0)
    time.sleep(2)

    last_cmd = ""
    last_cmd_mono = 0.0
    min_repeat_ms = float(os.getenv("ROBOT_CMD_MIN_REPEAT_MS", "0"))

    def send_cmd(cmd: str) -> None:
        nonlocal last_cmd, last_cmd_mono
        now = time.monotonic()
        min_dt = min_repeat_ms / 1000.0
        if cmd == last_cmd and min_dt > 0 and (now - last_cmd_mono) < min_dt:
            return
        last_cmd = cmd
        last_cmd_mono = now
        arduino.write((cmd + "\n").encode())
        arduino.flush()

    send_cmd("P:90")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(os.getenv("CAMERA_BUFFER_SIZE", "1")))

    grab_flush = max(0, int(os.getenv("CAMERA_GRAB_FLUSH", "1")))
    session = requests.Session()
    url = f"{args.server}/follow/step"
    preview = not args.no_preview and not _env_bool("HEADLESS", False)
    if preview:
        cv2.namedWindow("Pi follow client", cv2.WINDOW_NORMAL)

    encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, args.jpeg_quality))]
    post_interval = max(0.0, float(args.post_interval))

    def maybe_post_throttle() -> None:
        if post_interval > 0:
            time.sleep(post_interval)

    try:
        while True:
            for _ in range(grab_flush):
                cap.grab()
            ok, frame = cap.read()
            if not ok:
                break

            ok_buf, buf = cv2.imencode(".jpg", frame, encode_params)
            if not ok_buf:
                send_cmd("S:0")
                maybe_post_throttle()
                continue

            try:
                r = session.post(
                    url,
                    files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
                    timeout=args.timeout,
                )
            except requests.RequestException:
                send_cmd("S:0")
                if preview:
                    cv2.putText(
                        frame,
                        "NETWORK ERROR",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                    cv2.imshow("Pi follow client", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                time.sleep(0.05)
                maybe_post_throttle()
                continue

            if r.status_code != 200:
                send_cmd("S:0")
                maybe_post_throttle()
                continue

            data = r.json()
            if not data.get("ok") or "cmd" not in data:
                send_cmd("S:0")
                maybe_post_throttle()
                continue

            send_cmd(str(data["cmd"]))

            if preview:
                st = data.get("status", "")
                cv2.putText(
                    frame,
                    f"cmd: {data['cmd']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"status: {st}",
                    (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )
                bb = data.get("bbox")
                if bb and len(bb) == 4:
                    x1, y1, x2, y2 = map(int, bb)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"rt: {data.get('total_ms', 0):.0f}ms",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )
                cv2.imshow("Pi follow client", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            maybe_post_throttle()

    finally:
        send_cmd("S:0")
        cap.release()
        if preview:
            cv2.destroyAllWindows()
        arduino.close()


if __name__ == "__main__":
    main()
