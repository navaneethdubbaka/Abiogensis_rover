import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
FRAME_SIZE = 1280

def callback(indata, frames, time, status):
    audio = np.frombuffer(indata, dtype=np.int16)
    energy = np.abs(audio).mean()
    print(f"Energy: {energy:.1f}")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    blocksize=FRAME_SIZE,
    dtype="int16",
    channels=1,
    callback=callback,
):
    print("??? Speak loudly into the mic...")
    while True:
        sd.sleep(1000)
