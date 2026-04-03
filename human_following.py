import os
import glob
import cv2
import time
import math
import serial
from typing import Any, List, Optional, Tuple

from ultralytics import YOLO

# =========================
# CONFIG
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


def _parse_classes_filter() -> Optional[List[int]]:
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

TARGET_CLASS = "person"

FRAME_W = 640
FRAME_H = 480

FORWARD_SPEED = 150
TURN_SPEED_MIN = 115
TURN_SPEED_MAX = 200

CENTER_TOLERANCE = 0.05
STOP_AREA_RATIO = 0.35
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.4"))

LOOP_DELAY = float(os.getenv("LOOP_DELAY", "0"))
TARGET_LOOP_HZ = float(os.getenv("TARGET_LOOP_HZ", "0"))

LOST_FRAMES_BEFORE_SEARCH = int(os.getenv("LOST_FRAMES_BEFORE_SEARCH", "20"))
SEARCH_TURN_DURATION = float(os.getenv("SEARCH_TURN_DURATION", "0.25"))
SEARCH_ALTERNATE_EVERY = int(os.getenv("SEARCH_ALTERNATE_EVERY", "3"))

SMOOTH_ALPHA = float(os.getenv("X_DEV_SMOOTH_ALPHA", "0.35"))

ROBOT_YOLO_IMGSZ = int(os.getenv("ROBOT_YOLO_IMGSZ", "320"))
ROBOT_YOLO_DEVICE = os.getenv("ROBOT_YOLO_DEVICE", "").strip() or None
ROBOT_YOLO_HALF = _env_bool("ROBOT_YOLO_HALF", False)
ROBOT_YOLO_CLASSES = _parse_classes_filter()
USE_TRACKER = _env_bool("ROBOT_USE_TRACKER", True)
TRACKER_CFG = os.getenv("ROBOT_TRACKER", "bytetrack.yaml").strip() or "bytetrack.yaml"
YOLO_EVERY_N = max(1, int(os.getenv("YOLO_EVERY_N", "1")))
LOG_INFER_MS = _env_bool("LOG_INFER_MS", False)
LOG_INFER_EVERY = max(1, int(os.getenv("LOG_INFER_EVERY", "30")))

CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", "0"))
CAMERA_BUFFER_SIZE = int(os.getenv("CAMERA_BUFFER_SIZE", "1"))
CAMERA_GRAB_FLUSH = max(0, int(os.getenv("CAMERA_GRAB_FLUSH", "1")))

ROBOT_CMD_MIN_REPEAT_MS = float(os.getenv("ROBOT_CMD_MIN_REPEAT_MS", "0"))

TURN_PULSE_SEC = float(os.getenv("TURN_PULSE_SEC", "0"))


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


def box_center_xyxy(box: Any) -> Tuple[float, float]:
    if hasattr(box, "cpu"):
        box = box.cpu().numpy()
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def xyxy_to_int(box: Any) -> Tuple[int, int, int, int]:
    if hasattr(box, "cpu"):
        box = box.cpu().numpy()
    flat = box.reshape(-1) if hasattr(box, "reshape") else box
    x1, y1, x2, y2 = (int(float(flat[i])) for i in range(4))
    return x1, y1, x2, y2


def dist2(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def select_person_box(
    results,
    model_names,
    prev_track_id: Optional[int],
    prev_center: Optional[Tuple[float, float]],
):
    """
    Prefer previous track id, else closest box center to prev_center, else highest conf.
    Returns (box, conf, center_xy, track_id_or_none).
    """
    candidates: List[Tuple[Any, float, Optional[int]]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes = r.boxes
        xyxy = boxes.xyxy
        confs = boxes.conf
        clss = boxes.cls
        tid_tensor = getattr(boxes, "id", None)
        if tid_tensor is not None:
            tids = tid_tensor.cpu().numpy().astype(int).tolist()
        else:
            tids = [None] * len(boxes)
        for i in range(len(boxes)):
            c = float(confs[i])
            ci = int(clss[i])
            label = model_names[ci]
            if label != TARGET_CLASS or c < MIN_CONFIDENCE:
                continue
            box = xyxy[i]
            tid = tids[i] if i < len(tids) else None
            candidates.append((box, c, tid))

    if not candidates:
        return None, None, None, None

    if prev_track_id is not None:
        for box, c, tid in candidates:
            if tid is not None and int(tid) == int(prev_track_id):
                cx, cy = box_center_xyxy(box)
                return box, float(c), (cx, cy), int(tid)

    if prev_center is not None:
        best_box, best_c, best_tid = None, -1.0, None
        best_d = math.inf
        for box, c, tid in candidates:
            cx, cy = box_center_xyxy(box)
            d = dist2((cx, cy), prev_center)
            if d < best_d:
                best_d = d
                best_box, best_c, best_tid = box, c, tid
        if best_box is not None:
            cx, cy = box_center_xyxy(best_box)
            tid_i = int(best_tid) if best_tid is not None else None
            return best_box, float(best_c), (cx, cy), tid_i

    best_box, best_c, best_tid = None, -1.0, None
    for box, c, tid in candidates:
        if c > best_c:
            best_c = c
            best_box = box
            best_tid = tid
    cx, cy = box_center_xyxy(best_box)
    tid_i = int(best_tid) if best_tid is not None else None
    return best_box, float(best_c), (cx, cy), tid_i


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
status_text = "SEARCHING"
lost_frames = 0
search_phase = 0
x_dev_smooth = 0.0
prev_track_id: Optional[int] = None
prev_center: Optional[Tuple[float, float]] = None

search_active_until = 0.0
search_current_cmd: Optional[str] = None

frame_idx = 0
infer_count = 0
last_stale_box: Optional[Any] = None
last_stale_conf = 0.0

while True:
    loop_t0 = time.perf_counter()
    ret, frame = grab_frame()
    if not ret:
        break

    fh, fw = frame.shape[0], frame.shape[1]
    frame_area = float(fw * fh)

    frame_idx += 1
    run_infer = (frame_idx % YOLO_EVERY_N == 0)
    infer_ms = 0.0

    if run_infer:
        results, infer_ms = run_vision(frame, USE_TRACKER and not _tracker_failed)
        infer_count += 1
        best_box, best_conf, _ctr_pick, tid_pick = select_person_box(
            results, model.names, prev_track_id, prev_center
        )
        if best_box is not None:
            last_stale_box = best_box
            last_stale_conf = best_conf
        else:
            last_stale_box = None
            last_stale_conf = 0.0

        if LOG_INFER_MS and infer_count % LOG_INFER_EVERY == 0:
            mode = "track" if USE_TRACKER and not _tracker_failed else "predict"
            print(f"[human_following] YOLO {mode} infer: {infer_ms:.1f} ms")
    else:
        best_box = last_stale_box
        best_conf = last_stale_conf
        tid_pick = prev_track_id

    person_found = best_box is not None
    now = time.monotonic()

    if person_found:
        lost_frames = 0
        search_active_until = 0.0
        search_current_cmd = None
        x1, y1, x2, y2 = xyxy_to_int(best_box)

        if run_infer and tid_pick is not None:
            prev_track_id = tid_pick
        prev_center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {best_conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        box_width = x2 - x1
        box_height = y2 - y1
        box_area = box_width * box_height
        area_ratio = box_area / frame_area

        obj_center_x = (x1 + x2) / 2
        x_deviation = (obj_center_x / fw) - 0.5
        x_dev_smooth = SMOOTH_ALPHA * x_deviation + (1.0 - SMOOTH_ALPHA) * x_dev_smooth
        x_use = x_dev_smooth

        if abs(x_use) > CENTER_TOLERANCE:
            turn_strength = min(abs(x_use) * 400, TURN_SPEED_MAX)
            turn_speed = max(int(turn_strength), TURN_SPEED_MIN)
            cmd_turn = f"L:{turn_speed}" if x_use < 0 else f"R:{turn_speed}"

            if TURN_PULSE_SEC <= 0:
                send_cmd(cmd_turn)
                status_text = "TURN LEFT" if x_use < 0 else "TURN RIGHT"
            else:
                period = TURN_PULSE_SEC * 2.0
                phase_t = now % period if period > 0 else 0.0
                if phase_t < TURN_PULSE_SEC:
                    send_cmd(cmd_turn)
                    status_text = "TURN LEFT" if x_use < 0 else "TURN RIGHT"
                else:
                    send_cmd("S:0")
                    status_text = "TURN PAUSE"
        else:
            if area_ratio < STOP_AREA_RATIO:
                send_cmd(f"F:{FORWARD_SPEED}")
                status_text = "FORWARD"
            else:
                send_cmd("S:0")
                status_text = "STOP (1m reached)"

        cv2.putText(
            frame,
            f"x_dev: {x_use:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"area: {area_ratio:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

    else:
        prev_track_id = None
        prev_center = None

        in_search_pulse = now < search_active_until

        if not in_search_pulse:
            lost_frames += 1
            send_cmd("S:0")

        if in_search_pulse and search_current_cmd:
            send_cmd(search_current_cmd)
            status_text = "SEARCHING (scan)"
        elif lost_frames >= LOST_FRAMES_BEFORE_SEARCH:
            direction = "left" if (search_phase // SEARCH_ALTERNATE_EVERY) % 2 == 0 else "right"
            speed = TURN_SPEED_MIN
            search_current_cmd = f"L:{speed}" if direction == "left" else f"R:{speed}"
            send_cmd(search_current_cmd)
            search_active_until = now + SEARCH_TURN_DURATION
            search_phase += 1
            lost_frames = 0
            status_text = "SEARCHING (scan)"
        else:
            status_text = "SEARCHING"

    cv2.putText(
        frame,
        f"STATUS: {status_text}",
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
