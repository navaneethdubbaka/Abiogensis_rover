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
TURN_SPEED = 170

NEAR_AREA_THRESHOLD = 0.25   # stop when person fills ~25% of frame

SCAN_ROTATE_TIME = 0.6      # seconds to rotate for left/right scan
ROTATE_180_TIME = 1.0       # adjust for your robot

CAPTURE_SETTLE_TIME = 0.4   # wait after stopping before capture

# =========================
# SERIAL
# =========================
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

def send_cmd(cmd):
    arduino.write((cmd + "\n").encode())
    time.sleep(0.05)

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

# =========================
# DETECTION FUNCTION
# =========================
def detect_person(frame):
    results = model(frame, verbose=False)

    best_area = 0
    best_box = None

    for r in results:
        for box, conf, cls in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls):
            label = model.names[int(cls)]
            if label == TARGET_CLASS:
                x1, y1, x2, y2 = map(int, box)
                area = (x2 - x1) * (y2 - y1)

                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)

    return best_box, best_area

# =========================
# CAPTURE A CLEAN FRAME
# =========================
def capture_clean_frame():
    send_cmd("S")
    time.sleep(CAPTURE_SETTLE_TIME)

    # throw away a few frames to flush camera buffer
    for _ in range(3):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        return None

    return frame

# =========================
# STRICT SCAN SEQUENCE
# =========================
def scan_views_strict():
    views = {}

    # -------- FRONT --------
    frame = capture_clean_frame()
    if frame is not None:
        box, area = detect_person(frame)
        views["FRONT"] = (box, area)

    # -------- LEFT --------
    send_cmd(f"L:{TURN_SPEED}")
    time.sleep(SCAN_ROTATE_TIME)
    send_cmd("S")

    frame = capture_clean_frame()
    if frame is not None:
        box, area = detect_person(frame)
        views["LEFT"] = (box, area)

    # -------- RIGHT --------
    send_cmd(f"R:{TURN_SPEED}")
    time.sleep(SCAN_ROTATE_TIME * 2)
    send_cmd("S")

    frame = capture_clean_frame()
    if frame is not None:
        box, area = detect_person(frame)
        views["RIGHT"] = (box, area)

    # -------- RETURN TO FRONT --------
    send_cmd(f"L:{TURN_SPEED}")
    time.sleep(SCAN_ROTATE_TIME)
    send_cmd("S")

    return views

# =========================
# MAIN LOOP
# =========================
print("Autonomous follower started...")

while True:

    views = scan_views_strict()

    # Find best view
    best_dir = None
    best_area = 0
    best_box = None

    for direction, (box, area) in views.items():
        if box is not None and area > best_area:
            best_area = area
            best_box = box
            best_dir = direction

    # ===== NOTHING FOUND =====
    if best_dir is None:
        print("No person found. Rotating 180...")
        send_cmd("RB")
        time.sleep(ROTATE_180_TIME)
        continue

    print(f"Person found in {best_dir} | area={best_area}")

    # ===== TURN TOWARD TARGET =====
    if best_dir == "LEFT":
        send_cmd(f"L:{TURN_SPEED}")
        time.sleep(SCAN_ROTATE_TIME)
        send_cmd("S")

    elif best_dir == "RIGHT":
        send_cmd(f"R:{TURN_SPEED}")
        time.sleep(SCAN_ROTATE_TIME)
        send_cmd("S")

    # ===== APPROACH LOOP =====
    while True:
        frame = capture_clean_frame()
        if frame is None:
            break

        box, area = detect_person(frame)

        if box is None:
            print("Lost target. Breaking to rescan.")
            send_cmd("S")
            break

        x1, y1, x2, y2 = box

        # Debug draw
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.imshow("Follower", frame)
        cv2.waitKey(1)

        # Stop if near
        if area / (FRAME_W * FRAME_H) > NEAR_AREA_THRESHOLD:
            print("Reached target.")
            send_cmd("S")
            time.sleep(1)
            break

        # Move forward a short step
        send_cmd(f"F:{BASE_SPEED}")
        time.sleep(0.35)
        send_cmd("S")

        time.sleep(0.2)

    time.sleep(0.5)

# =========================
# CLEANUP
# =========================
cap.release()
cv2.destroyAllWindows()
send_cmd("S")
arduino.close()
