# -*- coding: utf-8 -*-

import numpy as np
import sounddevice as sd
import time
from openwakeword.model import Model

MODEL_PATH = "/home/pi/Desktop/sonic/sonic.onnx"

SAMPLE_RATE = 16000
FRAME_SIZE = 512
BUFFER_FRAMES = 8
THRESHOLD = 0.22
COOLDOWN_SECONDS = 2.0

# Use PulseAudio default (Pi mic)
sd.default.device = ("default", None)

model = Model(wakeword_model_paths=[MODEL_PATH])

def wait_for_wakeword():
    """
    Blocks until wake word 'sonic' is detected.
    Returns only after detection.
    """
    audio_buffer = np.zeros(FRAME_SIZE * BUFFER_FRAMES, dtype=np.int16)
    last_trigger_time = 0.0
    prev_score = 0.0
    detected = False

    def audio_callback(indata, frames, time_info, status):
        nonlocal audio_buffer, last_trigger_time, prev_score, detected

        audio_float = indata[:, 0]
        audio_int16 = np.clip(audio_float * 32768, -32768, 32767).astype(np.int16)

        # Sliding window
        audio_buffer[:-FRAME_SIZE] = audio_buffer[FRAME_SIZE:]
        audio_buffer[-FRAME_SIZE:] = audio_int16

        scores = model.predict(audio_buffer)

        for score in scores.values():
            now = time.time()
            if (
                prev_score < THRESHOLD
                and score >= THRESHOLD
                and (now - last_trigger_time) > COOLDOWN_SECONDS
            ):
                last_trigger_time = now
                detected = True
                raise sd.CallbackStop()

            prev_score = score

    # Block here until wake word is detected
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SIZE,
            channels=1,
            dtype="float32",
            callback=audio_callback,
        ):
            while not detected:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    print("?? Wake word detected: SONIC")
