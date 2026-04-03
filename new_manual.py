import sys, termios, tty, time, serial, select

SERIAL_PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"
BAUD = 115200

FORWARD_SPEED = 160
TURN_SPEED = 140

arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=0)

def send(cmd):
    arduino.write((cmd + "\n").encode())

def get_key():
    dr, _, _ = select.select([sys.stdin], [], [], 0)
    if dr:
        return sys.stdin.read(1)
    return None

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setcbreak(fd)

try:
    while True:
        key = get_key()

        if key:
            k = key.lower()

            if k == 'w':
                send(f"F:{FORWARD_SPEED}")
            elif k == 's':
                send(f"B:{FORWARD_SPEED}")
            elif k == 'a':
                send(f"L:{TURN_SPEED}")
            elif k == 'd':
                send(f"R:{TURN_SPEED}")
            elif k == 'q':
                send("S:0")
                break

        time.sleep(0.05)   # smooth control loop

finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
    arduino.close()
