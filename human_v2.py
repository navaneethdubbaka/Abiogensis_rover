import cv2
import time
import serial
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "yolo26n.pt"
SERIAL_PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"
BAUD = 115200

TARGET_CLASS = "person"

FRAME_W = 640
FRAME_H = 480

BASE_SPEED = 150
MAX_TURN_SPEED = 220
MIN_TURN_SPEED = 120

CENTER_TOLERANCE = 0.05
STOP_DISTANCE_Y = 0.15

# =========================
# SERIAL
# =========================
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

def send_cmd(cmd):
    arduino.write((cmd + "\n").encode())

# =========================
# PID
# =========================
class PID:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output

# Robot turn PID (tuned for 2 motors)
turn_pid = PID(Kp=2.2, Ki=0.0, Kd=0.35)

last_time = time.time()

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

cv2.namedWindow("Human Follower", cv2.WINDOW_NORMAL)

# =========================
# MOVEMENT LOGIC
# =========================
def move_robot_pid(x_deviation, y_max):
    global last_time

    now = time.time()
    dt = now - last_time
    last_time = now

    turn = turn_pid.compute(x_deviation, dt)

    TURN_SCALE = 320
    turn_speed = abs(turn) * TURN_SCALE
    turn_speed = max(min(turn_speed, MAX_TURN_SPEED), MIN_TURN_SPEED)

    # ----- FORWARD / STOP -----
    if abs(x_deviation) < CENTER_TOLERANCE:
        if (1 - y_max) < STOP_DISTANCE_Y:
            send_cmd("S")
            status = "STOP"
        else:
            send_cmd(f"F:{BASE_SPEED}")
            status = "FORWARD"

    # ----- TURN -----
    else:
        if turn > 0:
            send_cmd(f"L:{int(turn_speed)}")
            status = "LEFT"
        else:
            send_cmd(f"R:{int(turn_speed)}")
            status = "RIGHT"

    return status

# =========================
# MAIN LOOP
# =========================
fps_start = time.time()
frame_count = 0
status_text = "IDLE"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False)

    person_found = False
    best_box = None
    best_conf = 0

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            label = model.names[int(cls)]
            score = float(conf)

            if label == TARGET_CLASS and score > best_conf:
                best_conf = score
                best_box = box
                person_found = True

    if person_found:
        x1, y1, x2, y2 = map(int, best_box)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"person {best_conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Compute normalized deviation
        obj_x_center = (x1 + x2) / 2
        x_deviation = 0.5 - (obj_x_center / FRAME_W)
        y_max = y2 / FRAME_H

        # Robot control
        status_text = move_robot_pid(x_deviation, y_max)

        # Debug overlays
        cv2.putText(frame, f"x_dev: {x_deviation:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    else:
        send_cmd("S")
        status_text = "SEARCHING"

    # FPS calc
    frame_count += 1
    if frame_count >= 10:
        fps = frame_count / (time.time() - fps_start)
        fps_start = time.time()
        frame_count = 0
    else:
        fps = 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.putText(frame, f"STATUS: {status_text}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    cv2.imshow("Human Follower", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
send_cmd("S")
arduino.close()