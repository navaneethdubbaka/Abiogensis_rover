import json
import wave
from vosk import Model, KaldiRecognizer

MODEL_PATH = "models/vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

def speech_to_text(wav_file):
    wf = wave.open(wav_file, "rb")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        rec.AcceptWaveform(data)

    result = json.loads(rec.FinalResult())
    text = result.get("text", "").strip()

    print(f"?? You said: {text}")
    return text
