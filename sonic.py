from wakeword import wait_for_wakeword
from audio_utils import record_command
from stt import speech_to_text
from brain import route_command
from actions import move_robot, set_reminder

print("?? Sonic assistant online")

while True:
    # 1?? Wait for "sonic"
    wait_for_wakeword()

    # 2?? Record user command
    wav = record_command()

    # 3?? Speech ? text
    text = speech_to_text(wav)

    # 4?? Route intent via GPT-4o-mini
    result = route_command(text)

    # 5?? Execute action
    if result["intent"] == "MOVE":
        move_robot(result["action"])

    elif result["intent"] == "REMINDER":
        set_reminder(result["task"], result["time"])

    elif result["intent"] == "CHAT":
        print("?? Sonic:", result["reply"])
