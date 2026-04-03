import sounddevice as sd
import numpy as np
import time
import wave

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 300
SILENCE_SECONDS = 1.2
MAX_SECONDS = 6

def record_command(filename="command.wav"):
    print("?? Listening for command...")

    frames = []
    silence_start = None
    start_time = time.time()

    def callback(indata, frames_count, time_info, status):
        nonlocal silence_start
        audio = (indata[:, 0] * 32768).astype(np.int16)
        frames.append(audio)

        energy = np.abs(audio).mean()

        if energy < SILENCE_THRESHOLD:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start > SILENCE_SECONDS:
                raise sd.CallbackStop()
        else:
            silence_start = None

        if time.time() - start_time > MAX_SECONDS:
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while True:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    audio = np.concatenate(frames)

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print("?? Recording stopped")
    return filename
