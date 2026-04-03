import sys
import termios
import tty
import time
import serial

# =========================
# CONFIG
# =========================
SERIAL_PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"
BAUD = 115200

FORWARD_SPEED = 150
TURN_SPEED = 120

# =========================
# SERIAL
# =========================
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=1)
time.sleep(2)

def send_cmd(cmd):
    arduino.write((cmd + "\n").encode())
    print(f"Sent: {cmd}")

# =========================
# KEYBOARD INPUT
# =========================
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

print("""
=========================
MANUAL ROBOT CONTROL
=========================
W â†’ Forward
A â†’ Left
S â†’ Stop
D â†’ Right
R â†’ Rotate 180Â°
Q / ESC â†’ Quit
=========================
""")

try:
    while True:
        key = getch()

        if key.lower() == "w":
            send_cmd(f"F:{FORWARD_SPEED}")

        elif key.lower() == "a":
            send_cmd(f"L:{TURN_SPEED}")

        elif key.lower() == "d":
            send_cmd(f"R:{TURN_SPEED}")

        elif key.lower() == "s":
            send_cmd("S:0")

        elif key.lower() == "r":
            send_cmd("RB")

        elif key.lower() == "q" or ord(key) == 27:
            send_cmd("S:0")
            print("Exiting...")
            break

except KeyboardInterrupt:
    send_cmd("S:0")

finally:
    arduino.close()