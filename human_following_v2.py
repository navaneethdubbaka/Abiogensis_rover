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

FRAME_W = 640
FRAME_H = 480

FORWARD_SPEED = 150
TURN_SPEED_SLOW = 120
TURN_SPEED_FAST = 180

CENTER_TOLERANCE = 0.05
STOP_AREA_RATIO = 0.35
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.4"))

SERVO_CENTER = 90
SERVO_MIN = 35
SERVO_MAX = 145
SERVO_STEP = 2

STABLE_TIME = 2.0
LOOP_DELAY = 0.03

# Timed body turns (seconds) — avoid holding motor every frame
ALIGN_PULSE = float(os.getenv("ALIGN_PULSE_SEC", "0.12"))
RECOVER_PULSE = float(os.getenv("RECOVER_PULSE_SEC", "0.22"))

LOST_FRAMES_BEFORE_SWEEP = int(os.getenv("LOST_FRAMES_BEFORE_SWEEP", "25"))
SERVO_SWEEP_STEP = int(os.getenv("SERVO_SWEEP_STEP", "15"))
SERVO_SWEEP_PAUSE = float(os.getenv("SERVO_SWEEP_PAUSE", "0.08"))

# =========================
# SERIAL
# =========================
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
time.sleep(2)


def send(cmd: str):
    arduino.write((cmd + "\n").encode())
    arduino.flush()


def timed_turn(direction: str, speed: int, duration: float):
    """Send turn, wait, then stop (Pi / Arduino friendly)."""
    if direction == "left":
        send(f"L:{speed}")
    elif direction == "right":
        send(f"R:{speed}")
    time.sleep(duration)
    send("S:0")
    time.sleep(0.05)


servo_angle = SERVO_CENTER
send("P:90")

# =========================
# YOLO + CAMERA
# =========================
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

cv2.namedWindow("Human Following", cv2.WINDOW_NORMAL)

# =========================
# STATES
# =========================
TRACK_SERVO = 0
STABLE_LOCK = 1
ALIGN_BODY = 2
RECOVER_LIMIT = 3

state = TRACK_SERVO
stable_start = None
status = "SEARCHING"
lost_frames = 0


def run_servo_sweep_recovery():
    """Pan camera min->max->center to re-acquire subject."""
    global servo_angle
    send("S:0")
    for angle in range(SERVO_MIN, SERVO_MAX + 1, SERVO_SWEEP_STEP):
        servo_angle = min(SERVO_MAX, angle)
        send(f"P:{servo_angle}")
        time.sleep(SERVO_SWEEP_PAUSE)
    for angle in range(SERVO_MAX, SERVO_MIN - 1, -SERVO_SWEEP_STEP):
        servo_angle = max(SERVO_MIN, angle)
        send(f"P:{servo_angle}")
        time.sleep(SERVO_SWEEP_PAUSE)
    servo_angle = SERVO_CENTER
    send(f"P:{servo_angle}")
    time.sleep(SERVO_SWEEP_PAUSE)


# =========================
# MAIN LOOP
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[0], frame.shape[1]
    frame_area = float(fw * fh)

    results = model(frame, verbose=False)

    person = None
    conf_best = 0.0

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            c = float(conf)
            if model.names[int(cls)] == "person" and c >= MIN_CONFIDENCE and c > conf_best:
                conf_best = c
                person = box

    if person is None:
        lost_frames += 1
        send("S:0")
        state = TRACK_SERVO
        stable_start = None
        status = "SEARCHING"
        if lost_frames >= LOST_FRAMES_BEFORE_SWEEP:
            status = "SWEEP RECOVER"
            run_servo_sweep_recovery()
            lost_frames = 0
    else:
        lost_frames = 0
        x1, y1, x2, y2 = map(int, person)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {conf_best:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
        x_center = (x1 + x2) / 2
        x_dev = (x_center / fw) - 0.5

        if state == TRACK_SERVO:
            status = "SERVO TRACKING"

            if abs(x_dev) > CENTER_TOLERANCE:
                servo_angle += SERVO_STEP if x_dev < 0 else -SERVO_STEP
                servo_angle = max(SERVO_MIN, min(SERVO_MAX, servo_angle))
                send(f"P:{servo_angle}")
                stable_start = None
            else:
                state = STABLE_LOCK
                stable_start = time.time()

            if servo_angle <= SERVO_MIN + 2 or servo_angle >= SERVO_MAX - 2:
                state = RECOVER_LIMIT

        elif state == STABLE_LOCK:
            status = "HUMAN LOCKED"

            if abs(x_dev) > CENTER_TOLERANCE:
                state = TRACK_SERVO
                stable_start = None
            elif time.time() - stable_start >= STABLE_TIME:
                state = ALIGN_BODY

        elif state == ALIGN_BODY:
            status = "BODY ALIGNING"

            if servo_angle > SERVO_CENTER + 2:
                timed_turn("right", TURN_SPEED_SLOW, ALIGN_PULSE)
                servo_angle -= SERVO_STEP
                servo_angle = max(SERVO_MIN, servo_angle)
                send(f"P:{servo_angle}")

            elif servo_angle < SERVO_CENTER - 2:
                timed_turn("left", TURN_SPEED_SLOW, ALIGN_PULSE)
                servo_angle += SERVO_STEP
                servo_angle = min(SERVO_MAX, servo_angle)
                send(f"P:{servo_angle}")

            else:
                send("S:0")
                state = TRACK_SERVO
                stable_start = None

        elif state == RECOVER_LIMIT:
            status = "LIMIT RECOVERY"

            if servo_angle >= SERVO_MAX - 2:
                timed_turn("right", TURN_SPEED_FAST, RECOVER_PULSE)
            elif servo_angle <= SERVO_MIN + 2:
                timed_turn("left", TURN_SPEED_FAST, RECOVER_PULSE)
            else:
                send("S:0")
                state = TRACK_SERVO

        if state == TRACK_SERVO and abs(x_dev) < CENTER_TOLERANCE:
            if area_ratio < STOP_AREA_RATIO:
                send(f"F:{FORWARD_SPEED}")
            else:
                send("S:0")

    cv2.putText(
        frame,
        f"STATUS: {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"SERVO: {servo_angle}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )

    cv2.imshow("Human Following", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    time.sleep(LOOP_DELAY)

# =========================
# CLEANUP
# =========================
send("S:0")
cap.release()
cv2.destroyAllWindows()
arduino.close()
