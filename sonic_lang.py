import os
import time
import json
import wave
import threading
import numpy as np
import sounddevice as sd
import serial
from dotenv import load_dotenv
from datetime import datetime, timedelta

from openwakeword.model import Model
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# =========================
# ENV
# =========================
load_dotenv(".env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing")

# =========================
# AUDIO CONFIG
# =========================
SAMPLE_RATE = 16000
FRAME_SIZE = 1280
MIC_DEVICE = 1
WAKE_THRESHOLD = 0.50
COOLDOWN = 1.5

WAKE_MODEL_PATH = "sonic.onnx"

# =========================
# ARDUINO CONFIG
# =========================
SERIAL_PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_24238313635351910130-if00"
BAUD = 115200

FORWARD_SPEED = 150
TURN_SPEED = 150

print("🔌 Connecting to Arduino...")
arduino = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
time.sleep(2)
print("✅ Arduino connected")

def send_cmd(cmd):
    arduino.write((cmd + "\n").encode())
    arduino.flush()

send_cmd("P:90")

# =========================
# WAKEWORD MODEL
# =========================
wake_model = Model(wakeword_model_paths=[WAKE_MODEL_PATH])
_last_trigger = 0.0
_prev_score = 0.0

# =========================
# AUDIO UTILS
# =========================
def play_beep(freq=1000, duration=0.2):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = 0.5 * np.sin(2 * np.pi * freq * t)
    sd.play(tone, SAMPLE_RATE)
    sd.wait()

# =========================
import os

def speak(text):
    print("🗣️ Sonic says:", text)
    os.system(f'espeak "{text}" -s 120')  # -s speed



# =========================
# WAKE WORD LISTENER
# =========================
def wait_for_wakeword():
    global _last_trigger, _prev_score

    print("\n🎙️ Sonic listening for wake word...")
    detected = threading.Event()

    def callback(indata, frames, time_info, status):
        global _last_trigger, _prev_score

        audio = np.frombuffer(indata, dtype=np.int16)
        scores = wake_model.predict(audio)

        for score in scores.values():
            now = time.time()

            if _prev_score < WAKE_THRESHOLD and score >= WAKE_THRESHOLD:
                if now - _last_trigger > COOLDOWN:
                    _last_trigger = now
                    print(f"🔥 Wake word detected (score={score:.2f})")
                    detected.set()
                    return

            _prev_score = score

    with sd.InputStream(
        device=MIC_DEVICE,
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while not detected.is_set():
            sd.sleep(50)

    time.sleep(0.8)
    play_beep()
    time.sleep(0.8)

# =========================
# RECORD COMMAND
# =========================
def record_command(seconds=4):
    print("🎤 Listening for command...")
    audio = sd.rec(
        int(seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        device=MIC_DEVICE
    )
    sd.wait()

    path = "command.wav"
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    return path

# =========================
# SPEECH → TEXT
# =========================
def speech_to_text(path):
    client = OpenAI(api_key=OPENAI_API_KEY)
    with open(path, "rb") as f:
        result = client.audio.transcriptions.create(
            file=f,
            model="gpt-4o-mini-transcribe"
        )
    print("📝 You said:", result.text)
    return result.text

# =========================
# TOOLS
# =========================
@tool
def move_robot(action: str):
    """Move robot: forward, backward, left, right, stop"""
    print(f"🤖 Executing MOVE: {action}")

    if action == "forward":
        send_cmd(f"F:{FORWARD_SPEED}")
    elif action == "backward":
        send_cmd(f"B:{FORWARD_SPEED}")
    elif action == "left":
        send_cmd(f"L:{TURN_SPEED}")
    elif action == "right":
        send_cmd(f"R:{TURN_SPEED}")
    else:
        send_cmd("S:0")

    speak(f"Moving {action}")

@tool
def set_reminder(task: str, time: str):
    """Set a reminder"""
    speak(f"Reminder set for {task}")

@tool
def chat(reply: str):
    """Speak to the user"""
    speak(reply)

TOOLS = [move_robot, set_reminder, chat]

# =========================
# LLM (TOOLS MODE)
# =========================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
).bind_tools(TOOLS)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are Sonic, a robot assistant.\n"
     "Always call one tool.\n"
     "Never respond with plain text."
    ),
    ("human", "{input}")
])

# =========================
# MAIN LOOP
# =========================
print("\n🤖 Sonic assistant online")

while True:
    wait_for_wakeword()

    wav = record_command()
    text = speech_to_text(wav)

    print("🤖 Thinking...")
    response = llm.invoke(prompt.format_messages(input=text))

    if response.tool_calls:
        for call in response.tool_calls:
            for tool_fn in TOOLS:
                if tool_fn.name == call["name"]:
                    tool_fn.invoke(call["args"])
