import pvporcupine
import pyaudio
import struct
import subprocess
import sys
import os

ACCESS_KEY = "H/mpz5xrWhMNBteOVX2zebV8kLgVoRKYSHpvUPsxuXlr0WqkQziATw=="
WAKE_WORD_PATH = "sonic.ppn"

# Use venv python explicitly
VENV_PYTHON = os.path.join("venv", "bin", "python")

porcupine = pvporcupine.create(
    access_key=ACCESS_KEY,
    keywords=["jarvis","porcupine","bumblebee"]
)

pa = pyaudio.PyAudio()

stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("Listening for wake word (offline)...")

assistant_process = None  # prevent multiple launches

try:
    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm = struct.unpack_from(
            "h" * porcupine.frame_length,
            pcm
        )

        keyword_index = porcupine.process(pcm)

        if keyword_index >= 0:
            print("Wake word detected!")

            # Launch assistant only if not already running
            if assistant_process is None or assistant_process.poll() is not None:
                assistant_process = subprocess.Popen(
                    [VENV_PYTHON, "assistant.py"]
                )
            else:
                print("Assistant already running")

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    stream.close()
    pa.terminate()
    porcupine.delete()
