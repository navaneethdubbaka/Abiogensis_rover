import time
import serial
import threading
from datetime import datetime, timedelta

# =========================
# CONFIG
# =========================
SERIAL_PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"
BAUD = 115200

FORWARD_SPEED = 150
TURN_SPEED = 150

# =========================
# SERIAL INIT
# =========================
try:
    arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
    time.sleep(2)
except Exception as e:
    raise RuntimeError(f"Failed to connect to Arduino: {e}")

def send_cmd(cmd: str):
    arduino.write((cmd + "\n").encode())
    arduino.flush()

# Center servo once
send_cmd("P:90")

# =========================
# ROBOT ACTIONS
# =========================
def move_robot(action: str):
    action = action.lower()

    if action == "forward":
        send_cmd(f"F:{FORWARD_SPEED}")

    elif action == "backward":
        send_cmd(f"B:{FORWARD_SPEED}")

    elif action == "left":
        send_cmd(f"L:{TURN_SPEED}")

    elif action == "right":
        send_cmd(f"R:{TURN_SPEED}")

    elif action == "stop":
        send_cmd("S:0")

    else:
        print(f"?? Unknown move action: {action}")
        send_cmd("S:0")

# =========================
# REMINDER SYSTEM
# =========================
_reminders = []

def _parse_time(time_str: str) -> datetime:
    """
    Handles:
    - 'in 10 minutes'
    - 'in 2 hours'
    - 'in 30 seconds'
    """
    time_str = time_str.lower().strip()

    if time_str.startswith("in"):
        parts = time_str.split()
        amount = int(parts[1])
        unit = parts[2]

        if "second" in unit:
            return datetime.now() + timedelta(seconds=amount)
        elif "minute" in unit:
            return datetime.now() + timedelta(minutes=amount)
        elif "hour" in unit:
            return datetime.now() + timedelta(hours=amount)

    # fallback ? 1 minute
    return datetime.now() + timedelta(minutes=1)

def set_reminder(task: str, time_str: str):
    trigger_time = _parse_time(time_str)

    reminder = {
        "task": task,
        "time": trigger_time,
        "triggered": False
    }

    _reminders.append(reminder)
    print(f"? Reminder set: '{task}' at {trigger_time.strftime('%H:%M:%S')}")

def _reminder_loop():
    while True:
        now = datetime.now()

        for r in _reminders:
            if not r["triggered"] and now >= r["time"]:
                r["triggered"] = True
                print(f"\n?? REMINDER: {r['task']}\n")

        time.sleep(1)

# Start reminder thread
threading.Thread(target=_reminder_loop, daemon=True).start()

# =========================
# CLEANUP
# =========================
def cleanup():
    try:
        send_cmd("S:0")
        arduino.close()
    except:
        pass
