import os
import glob
import cv2
import time
import serial
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

LOOP_DELAY = 0.05

# Lost target: timed chassis nudge to search (no pan servo in v1)
LOST_FRAMES_BEFORE_SEARCH = int(os.getenv("LOST_FRAMES_BEFORE_SEARCH", "20"))
SEARCH_TURN_DURATION = float(os.getenv("SEARCH_TURN_DURATION", "0.25"))
SEARCH_ALTERNATE_EVERY = int(os.getenv("SEARCH_ALTERNATE_EVERY", "3"))

# Light smoothing on horizontal error to reduce twitching
SMOOTH_ALPHA = float(os.getenv("X_DEV_SMOOTH_ALPHA", "0.35"))

# =========================
# SERIAL
# =========================
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
time.sleep(2)


def send_cmd(cmd: str):
    arduino.write((cmd + "\n").encode())
    arduino.flush()


send_cmd("P:90")

# =========================
# YOLO
# =========================
model = YOLO(MODEL_PATH)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

cv2.namedWindow("Person Centering Robot", cv2.WINDOW_NORMAL)

# =========================
# MAIN LOOP
# =========================
status_text = "SEARCHING"
lost_frames = 0
search_phase = 0
x_dev_smooth = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[0], frame.shape[1]
    frame_area = float(fw * fh)

    results = model(frame, verbose=False)

    person_found = False
    best_box = None
    best_conf = 0.0

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            label = model.names[int(cls)]
            c = float(conf)
            if label == TARGET_CLASS and c >= MIN_CONFIDENCE and c > best_conf:
                best_conf = c
                best_box = box
                person_found = True

    if person_found:
        lost_frames = 0
        x1, y1, x2, y2 = map(int, best_box)

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

            if x_use < 0:
                send_cmd(f"L:{turn_speed}")
                status_text = "TURN LEFT"
            else:
                send_cmd(f"R:{turn_speed}")
                status_text = "TURN RIGHT"

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
        lost_frames += 1
        send_cmd("S:0")
        if lost_frames >= LOST_FRAMES_BEFORE_SEARCH:
            status_text = "SEARCHING (scan)"
            direction = "left" if (search_phase // SEARCH_ALTERNATE_EVERY) % 2 == 0 else "right"
            speed = TURN_SPEED_MIN
            if direction == "left":
                send_cmd(f"L:{speed}")
            else:
                send_cmd(f"R:{speed}")
            time.sleep(SEARCH_TURN_DURATION)
            send_cmd("S:0")
            search_phase += 1
            lost_frames = 0
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

    time.sleep(LOOP_DELAY)

# =========================
# CLEANUP
# =========================
send_cmd("S:0")
cap.release()
cv2.destroyAllWindows()
arduino.close()
