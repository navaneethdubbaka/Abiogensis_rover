import serial
import time

SERIAL_PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"

arduino = serial.Serial(SERIAL_PORT, 115200, timeout=1)
time.sleep(2)

print("Sending FORWARD")
arduino.write(b"F:140\n")
time.sleep(2)

print("Sending STOP")
arduino.write(b"S:0\n")

arduino.close()
