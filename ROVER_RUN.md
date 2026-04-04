# How to run the human-following rover

This project can run **vision and control on the Raspberry Pi only**, or **offload YOLO to a PC with a GPU** while the Pi handles the camera and Arduino serial.

---

## What you need

- **Raspberry Pi** (e.g. Pi 5) with camera (USB or CSI)
- **Arduino** (or compatible) driving motors, connected over USB serial
- Firmware that understands the same **serial commands** as this repo (e.g. `L:120`, `R:120`, `F:150`, `S:0`, `P:90`)
- **YOLO weights** (e.g. `yolo26n.pt`) on the machine that runs inference

---

## Serial port

| Platform | Typical port | Environment variable |
|----------|--------------|----------------------|
| Raspberry Pi | `/dev/ttyACM0` or `/dev/ttyUSB0` | `ROBOT_SERIAL_PORT` |
| Windows | `COM3` (varies) | `ROBOT_SERIAL_PORT=COM3` |

If the port is wrong, the script will fail at startup or the rover will not move.

---

## Mode A — Everything on the Raspberry Pi

YOLO runs on the Pi (CPU or Pi GPU if supported). No PC required.

1. Install dependencies (Pi), including **Ultralytics** and **OpenCV**:

   ```bash
   pip install ultralytics opencv-python pyserial numpy
   ```

2. Place **YOLO weights** in the repo folder or set:

   ```bash
   export YOLO_MODEL_PATH=/path/to/your/model.pt
   ```

3. From the repo root:

   ```bash
   export ROBOT_SERIAL_PORT=/dev/ttyACM0   # adjust
   python human_following.py
   ```

4. Press **Esc** in the OpenCV window to stop. The script sends `S:0` on exit.

Optional tuning uses the same variables as in [`pc_follow_server.env.example`](pc_follow_server.env.example) (follow policy + `ROBOT_YOLO_*`). Set them in the shell or a `.env` you load before starting—`human_following.py` reads **environment variables only** (it does not auto-load a `.env` file unless your shell does).

**Related files:** `human_following.py`, `follow_policy.py`

---

## Mode B — PC GPU inference, Pi camera + serial

The **PC** runs YOLO and the follow policy; the **Pi** captures JPEGs, POSTs them to the PC, and forwards the returned command to the Arduino.

### On the PC (laptop / desktop with NVIDIA GPU recommended)

1. Install PC dependencies:

   ```bash
   pip install -r requirements_pc_follow.txt
   ```

2. Copy and edit environment (optional but recommended):

   ```bash
   cp pc_follow_server.env.example .env
   # Edit .env: YOLO_MODEL_PATH, ROBOT_YOLO_DEVICE=0, PC LAN tuning, etc.
   ```

   Load `.env` in your terminal (or set variables in the IDE). Example for PowerShell is in the top comments of `pc_follow_server.env.example`.

3. Start the API server from the **repository root** (so imports and model paths resolve):

   ```bash
   uvicorn pc_follow_server:app --host 0.0.0.0 --port 8765
   ```

   Or:

   ```bash
   python pc_follow_server.py
   ```

4. Confirm health (from PC or Pi):

   ```bash
   curl http://127.0.0.1:8765/health
   ```

Use the PC’s **LAN IP** (not the router/gateway IP), e.g. `192.168.1.42`.

**Related files:** `pc_follow_server.py`, `follow_policy.py`, `pc_follow_server.env.example`, `requirements_pc_follow.txt`

### On the Raspberry Pi

1. Install (no Ultralytics needed for this mode):

   ```bash
   pip install opencv-python pyserial requests
   ```

2. Point the client at your PC:

   ```bash
   export ROBOT_SERIAL_PORT=/dev/ttyACM0
   export ROBOT_FOLLOW_SERVER=http://192.168.x.x:8765
   python pi_follow_client.py
   ```

   Or in one line:

   ```bash
   python pi_follow_client.py --server http://192.168.x.x:8765
   ```

3. On network errors or timeouts, the client sends **`S:0`** to stop motors.

**Ethernet** between Pi and LAN is more stable than Wi‑Fi for smooth following.

**Related files:** `pi_follow_client.py`

---

## Resetting follow state (PC mode)

If the tracker or “lost person” logic gets confused, restart the server or call:

```bash
curl -X POST http://PC_IP:8765/follow/reset
```

---

## Other scripts in the repo

- **`human_following_v2.py`** — Alternate local follow flow with servo-oriented states (run on the device that has camera + serial + YOLO if you use it that way).
- **`client_rpi.py`** — Older client that talks to a server exposing `/detect/realtime` (different API than `/follow/step`).

---

## Quick troubleshooting

| Symptom | Things to check |
|---------|------------------|
| No serial / permission denied (Pi) | User in `dialout` group; correct `ROBOT_SERIAL_PORT` |
| Windows serial | `ROBOT_SERIAL_PORT=COMx` |
| Pi cannot reach PC | Same subnet, ping PC IP, firewall allows TCP **8765** on the PC |
| Fast left–right wag | Wired LAN; tune `TURN_ENTER_TOLERANCE`, `X_DEV_SMOOTH_ALPHA`, `TURN_DIRECTION_DEBOUNCE_SEC`, `ALIGN_TURN_GAIN` on the PC (see `pc_follow_server.env.example`) |
| Model not found | `YOLO_MODEL_PATH` or place `yolo26n.pt` next to the scripts |

---

## Safety

Always keep a **physical e-stop** or power switch. Network and vision can fail; the Pi client tries to stop on errors, but that is not a substitute for safe hardware design.
