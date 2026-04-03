import os
import glob
import cv2
import time
import serial
from typing import Optional

from ultralytics import YOLO

from follow_policy import (
    FollowState,
    follow_config_from_env,
    follow_control_step,
    select_person_box,
)

# =========================
# CONFIG (YOLO / camera / serial only; follow tuning via follow_policy / env)
# =========================
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


def resolve_yolo_model_path() -> str:
    env = os.getenv("YOLO_MODEL_PATH")
    if env and env.strip():
        return env.strip()
    for p in (
        os.path.join(_SCRIPT_DIR, "yolo26n.pt"),
        os.path.join(_SCRIPT_DIR, "sonic_lang", "yolo26n.pt"),
    ):
        if os.path.isfile(p):
            return p
    return "yolo26n.pt"


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _parse_classes_filter():
    raw = os.getenv("ROBOT_YOLO_CLASSES", "0")
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s == "all":
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


SERIAL_PORT = resolve_robot_serial_port()
MODEL_PATH = resolve_yolo_model_path()
BAUD = int(os.getenv("ROBOT_BAUD_RATE", "115200"))

FRAME_W = 640
FRAME_H = 480

LOOP_DELAY = float(os.getenv("LOOP_DELAY", "0"))
TARGET_LOOP_HZ = float(os.getenv("TARGET_LOOP_HZ", "0"))

ROBOT_YOLO_IMGSZ = int(os.getenv("ROBOT_YOLO_IMGSZ", "320"))
ROBOT_YOLO_DEVICE = os.getenv("ROBOT_YOLO_DEVICE", "").strip() or None
ROBOT_YOLO_HALF = _env_bool("ROBOT_YOLO_HALF", False)
ROBOT_YOLO_CLASSES = _parse_classes_filter()
USE_TRACKER = _env_bool("ROBOT_USE_TRACKER", True)
TRACKER_CFG = os.getenv("ROBOT_TRACKER", "bytetrack.yaml").strip() or "bytetrack.yaml"
YOLO_EVERY_N = max(1, int(os.getenv("YOLO_EVERY_N", "1")))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.4"))
LOG_INFER_MS = _env_bool("LOG_INFER_MS", False)
LOG_INFER_EVERY = max(1, int(os.getenv("LOG_INFER_EVERY", "30")))

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_BUFFER_SIZE = int(os.getenv("CAMERA_BUFFER_SIZE", "1"))
CAMERA_GRAB_FLUSH = max(0, int(os.getenv("CAMERA_GRAB_FLUSH", "1")))

ROBOT_CMD_MIN_REPEAT_MS = float(os.getenv("ROBOT_CMD_MIN_REPEAT_MS", "0"))

_follow_cfg = follow_config_from_env()


def build_yolo_kwargs() -> dict:
    kw: dict = {
        "imgsz": ROBOT_YOLO_IMGSZ,
        "verbose": False,
        "conf": MIN_CONFIDENCE,
    }
    if ROBOT_YOLO_CLASSES is not None:
        kw["classes"] = ROBOT_YOLO_CLASSES
    if ROBOT_YOLO_DEVICE is not None:
        kw["device"] = ROBOT_YOLO_DEVICE
    if ROBOT_YOLO_HALF:
        kw["half"] = True
    return kw


YOLO_KWARGS = build_yolo_kwargs()

# =========================
# SERIAL
# =========================
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
time.sleep(2)

_last_cmd: str = ""
_last_cmd_mono: float = 0.0


def send_cmd(cmd: str) -> None:
    global _last_cmd, _last_cmd_mono
    now = time.monotonic()
    min_dt = ROBOT_CMD_MIN_REPEAT_MS / 1000.0
    if (
        cmd == _last_cmd
        and min_dt > 0
        and (now - _last_cmd_mono) < min_dt
    ):
        return
    _last_cmd = cmd
    _last_cmd_mono = now
    arduino.write((cmd + "\n").encode())
    arduino.flush()


send_cmd("P:90")

# =========================
# YOLO
# =========================
model = YOLO(MODEL_PATH)
_tracker_failed = False


def run_vision(frame, use_track: bool):
    global _tracker_failed
    t0 = time.perf_counter()
    if use_track and not _tracker_failed:
        try:
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                **YOLO_KWARGS,
            )
        except Exception:
            _tracker_failed = True
            results = model.predict(frame, **YOLO_KWARGS)
    else:
        results = model.predict(frame, **YOLO_KWARGS)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    return results, dt_ms


# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

cv2.namedWindow("Person Centering Robot", cv2.WINDOW_NORMAL)


def grab_frame():
    for _ in range(CAMERA_GRAB_FLUSH):
        cap.grab()
    return cap.read()


# =========================
# MAIN LOOP
# =========================
state = FollowState()
infer_count = 0

while True:
    loop_t0 = time.perf_counter()
    ret, frame = grab_frame()
    if not ret:
        break

    fh, fw = frame.shape[0], frame.shape[1]

    state.frame_idx += 1
    run_infer = (state.frame_idx % YOLO_EVERY_N == 0)
    infer_ms = 0.0

    if run_infer:
        results, infer_ms = run_vision(frame, USE_TRACKER and not _tracker_failed)
        infer_count += 1
        best_box, best_conf, _ctr_pick, tid_pick = select_person_box(
            results,
            model.names,
            _follow_cfg,
            state.prev_track_id,
            state.prev_center,
        )
        if best_box is not None:
            state.last_stale_box = best_box
            state.last_stale_conf = float(best_conf or 0.0)
        else:
            state.last_stale_box = None
            state.last_stale_conf = 0.0

        if LOG_INFER_MS and infer_count % LOG_INFER_EVERY == 0:
            mode = "track" if USE_TRACKER and not _tracker_failed else "predict"
            print(f"[human_following] YOLO {mode} infer: {infer_ms:.1f} ms")
    else:
        best_box = state.last_stale_box
        best_conf = state.last_stale_conf
        tid_pick: Optional[int] = state.prev_track_id

    person_found = best_box is not None
    now = time.monotonic()

    step = follow_control_step(
        state,
        _follow_cfg,
        now=now,
        fw=fw,
        fh=fh,
        person_found=person_found,
        best_box=best_box,
        best_conf=float(best_conf or 0.0),
        tid_pick=tid_pick if run_infer else state.prev_track_id,
        run_infer=run_infer,
    )

    send_cmd(step.cmd)

    if step.bbox is not None:
        x1, y1, x2, y2 = step.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {step.conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"x_dev: {step.x_dev_smooth:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"area: {step.area_ratio:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    cv2.putText(
        frame,
        f"STATUS: {step.status}",
        (10, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
    )

    cv2.imshow("Person Centering Robot", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    work_dt = time.perf_counter() - loop_t0
    if TARGET_LOOP_HZ > 0:
        target_period = 1.0 / TARGET_LOOP_HZ
        time.sleep(max(0.0, target_period - work_dt))
    elif LOOP_DELAY > 0:
        time.sleep(max(0.0, LOOP_DELAY - work_dt))

# =========================
# CLEANUP
# =========================
send_cmd("S:0")
cap.release()
cv2.destroyAllWindows()
arduino.close()
