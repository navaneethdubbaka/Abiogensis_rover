# Safety monitor — AI surveillance assistant (Ollama + Telegram)

This module runs on a **PC with a GPU** (recommended): it receives camera images (usually from a **Raspberry Pi** over the network), runs a **vision-language model** via **Ollama**, classifies situations as safe / warning / critical, writes **JSONL logs**, and can send **Telegram** alerts with the frame and a short rationale.

The Pi is intentionally **thin**: it only captures video and uploads JPEGs. It does **not** run Ollama or hold Telegram credentials.

> This repository also contains rover follow/stack code. For rover-specific operations, see [`ROVER_RUN.md`](../ROVER_RUN.md) in the repo root.

---

## Architecture

| Role | Machine | Responsibilities |
|------|---------|------------------|
| **Uploader** | Raspberry Pi running **Raspberry Pi OS** (or any client) | `OpenCV` capture → JPEG → `POST /safety/ingest` |
| **Brain** | PC | Ingest HTTP, static-scene gate, Ollama VLM, rules, logs, Telegram |

Ollama should listen on the **same PC** as the safety server (typically `http://127.0.0.1:11434`) so inference stays local and uses the GPU.

---

## Prerequisites

### PC

- Windows or Linux
- [Python](https://www.python.org/) 3.10+ recommended
- [Ollama](https://ollama.com/) installed and a **vision** model pulled, for example:
  - `ollama pull llava`  
  - or another model that supports images in `/api/chat` (check Ollama docs for your chosen model)
- Optional: USB webcam if you use **local** capture on the PC (`SAFETY_LOCAL_CAMERA_INDICES`)

### Raspberry Pi (Raspberry Pi OS)

These steps assume **Raspberry Pi OS** (Desktop or Lite; 64-bit Bookworm is typical). Other distros on Pi may work but paths (e.g. `apt` packages) can differ.

- Python 3 (`python3` / venv)
- USB webcam **or** official Camera Module — see [§6](#6-raspberry-pi-install-and-run-the-uploader) for which stack to use
- Network route to the PC (same LAN is typical)

---

## 1. Install Ollama and a vision model (PC)

1. Install Ollama from the official site. With **`SAFETY_AUTO_START_OLLAMA=1`** (default) and **`OLLAMA_BASE_URL`** pointing at `127.0.0.1` or `localhost`, the safety server runs **`ollama serve`** in the background if the API is not already up. Set **`SAFETY_AUTO_START_OLLAMA=0`** if you only use the Ollama app/service and do not want a second process.
2. Pull a model, e.g.:
   ```bash
   ollama pull llava
   ```
3. Confirm:
   ```bash
   ollama list
   ```

Set the model name later in `.env` as `SAFETY_VLM_MODEL` (e.g. `llava` or the exact tag Ollama shows).

---

## 2. Install Python dependencies (PC)

The file `requirements_safety.txt` lives in the **repository root** (parent of `safety_monitor/`).

From the **repository root**:

```bash
pip install -r requirements_safety.txt
```

If your shell is already inside `safety_monitor/`, use either:

```bash
pip install -r requirements.txt
```

(that file includes the root list) or:

```bash
pip install -r ../requirements_safety.txt
```

---

## 3. Configure environment (PC)

1. Copy the example env file to the **repo root** as `.env` (same place `python-dotenv` expects by default):

   ```bash
   copy safety_monitor.env.example .env
   ```
   (On Linux/macOS: `cp safety_monitor.env.example .env`.)

2. Edit `.env` and set at least:

   | Variable | Purpose |
   |----------|---------|
   | `SAFETY_VLM_MODEL` | Ollama vision model name (e.g. `llava`) |
   | `OLLAMA_BASE_URL` | Usually `http://127.0.0.1:11434` |
   | `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/BotFather) |
   | `TELEGRAM_CHAT_ID` | Your chat ID (see Telegram section below) |

3. Optional but useful:

   - `SAFETY_SERVER_PORT` — default `8766` (does not conflict with `pc_follow_server` on `8765`).
   - `SAFETY_INGEST_SECRET` — if set, every ingest must send header `X-Safety-Token: <same value>`.
   - `SAFETY_DRY_RUN=1` — log events and call Ollama but **do not** send Telegram/webhook (good for testing).
   - `SAFETY_RULES_FILE` — path to a YAML rules file (see [Rules](#rules-yaml)).
   - `SAFETY_CONFIG_YAML` — optional extra YAML merged with env (e.g. `camera_hints`).

Full variable list and comments: [`safety_monitor.env.example`](../safety_monitor.env.example).

---

## 4. Telegram setup (PC only)

1. Open Telegram, search **@BotFather**, run `/newbot`, and copy the **HTTP API token** → `TELEGRAM_BOT_TOKEN`.
2. Get your **chat ID**:
   - Message **@userinfobot** and read the `Id` field, or  
   - Start a chat with your bot, send any message, then open  
     `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates` in a browser and read `chat.id`.
3. Put that numeric ID in `TELEGRAM_CHAT_ID`.

`TELEGRAM_PARSE_MODE=HTML` is optional; if captions break because of special characters, clear it or escape HTML in operator text.

---

## 5. Run the safety server (PC)

From the **repository root**:

```bash
python -m safety_monitor.main
```

By default the API listens on `0.0.0.0:8766` (configurable via `SAFETY_SERVER_HOST` / `SAFETY_SERVER_PORT`).

### Health check

```bash
curl http://127.0.0.1:8766/safety/health
```

You should see `ollama_reachable`, `model_configured`, and `telegram_configured` reflecting your `.env`.

With **`SAFETY_PRINT_OLLAMA_RESPONSE=1`** (default), each Ollama assistant reply (or connection/parse error text) is printed to the **uvicorn terminal**, truncated by `SAFETY_OLLAMA_LOG_MAX_CHARS`. Set **`SAFETY_PRINT_OLLAMA_RESPONSE=0`** to turn that off.

### Alternative: Uvicorn directly

```bash
uvicorn safety_monitor.server:app --host 0.0.0.0 --port 8766
```

---

## 6. Raspberry Pi: install and run the uploader

Instructions below target **Raspberry Pi OS** (`apt`, V4L2 USB stack, optional `python3-opencv` from Raspberry Pi / Debian repos).

**USB webcam (default):** `safety_pi_sender.py` opens the camera like **`robot_listener.py`**: `cv2.VideoCapture(CAMERA_INDEX)`. **Live view** is **not** `imshow` (avoids Qt/GTK issues); it starts a small **MJPEG server** (FastAPI + uvicorn). Open **`http://127.0.0.1:9080/`** on the Pi (or `http://<pi-ip>:9080/` from a phone/PC on the LAN). Use `--no-preview` to skip that server.

**Official CSI / ribbon camera:** install `picamera2` and run with **`--picamera2`** or set **`SAFETY_PI_USE_PICAMERA2=1`**. Do **not** use that for a USB camera.

On the Pi, copy the repo (or at least `safety_pi_sender.py` and `requirements_safety_pi.txt`), then:

```bash
pip install -r requirements_safety_pi.txt
```

Set environment variables (or use a `.env` in the working directory if you installed `python-dotenv`):

| Variable | Example | Purpose |
|----------|---------|---------|
| `SAFETY_PC_BASE_URL` | `http://192.168.1.50:8766` | PC safety server URL (no trailing slash) |
| `SAFETY_CAMERA_ID` | `pi_cam` | Logical name on the PC (multi-camera) |
| `CAMERA_INDEX` | `0` | USB camera device index for OpenCV |
| `SAFETY_UPLOAD_INTERVAL_SEC` | `7.5` | Seconds between uploads |
| `SAFETY_INGEST_TOKEN` | same as PC `SAFETY_INGEST_SECRET` | If PC uses ingest secret |
| `SAFETY_PI_PREVIEW_PORT` | `9080` | Browser MJPEG preview (`http://<pi>:9080/`) |
| `SAFETY_PI_OPEN_BROWSER` | `1` | Open preview in default browser when `DISPLAY` is set |
| `SAFETY_PI_STREAM_JPEG_QUALITY` | `80` | JPEG quality for the MJPEG stream |
| `SAFETY_PI_USE_PICAMERA2` | `0` (default) | Set `1` only for **CSI** camera (or use `--picamera2`) |

Run (USB example):

```bash
python safety_pi_sender.py --server http://YOUR_PC_IP:8766 --camera-id pi_cam --camera 0
```

Useful flags: `--camera 0`, `--width 640`, `--height 480`, `--jpeg-quality 80`, `--interval 7.5`, `--timeout 15`, `--token <secret>`.

Preview uses **`pip install fastapi uvicorn`** (listed in `requirements_safety_pi.txt`), same idea as [`sonic_lang/robot_listener.py`](../sonic_lang/robot_listener.py) `/camera/stream`. No desktop OpenCV GUI required.

### Older note: OpenCV `imshow`

The sender **no longer** uses `cv2.imshow` for USB. If you only need uploads, use `--no-preview`.

---

## 7. HTTP API reference

### `POST /safety/ingest`

Multipart form:

- **`image`** (file): JPEG frame (required).
- **`camera_id`** (form field): string ID (default `pi_cam`).

Optional header: `X-Safety-Token` if `SAFETY_INGEST_SECRET` is set on the PC.

**Response:** `{"ok": true, "queued": true, "camera_id": "..."}`  
Processing is asynchronous; the server returns after enqueueing.

### `POST /safety/ingest/text`

Form fields:

- **`camera_id`**
- **`text`** — operator note or sensor context for that camera (merged into the next Ollama user message for that stream).

Same optional `X-Safety-Token` header.

### `GET /safety/health`

JSON status: Ollama reachability, whether model/Telegram are configured, `dry_run` flag.

---

## 8. Behavior summary

- **Static-scene gate:** If the downscaled grayscale image barely changes for several frames in a row, the server **skips** Ollama for that camera to save GPU time (tunable via `SKIP_VLM_ON_STATIC`, `SAFETY_STATIC_DIFF_THRESHOLD`, `SAFETY_STATIC_FRAMES_K`).
- **Notifications:** By default, **warning** and **critical** can trigger Telegram; **critical** is not debounced the same way as repeated **warnings** (see `SAFETY_WARNING_DEBOUNCE_SEC`). **Safe** notifications are off unless `NOTIFY_ON_SAFE=1`.
- **Logs:** Rotating JSONL under `SAFETY_LOG_DIR` (default `logs/safety_events.jsonl`).

---

## 9. Optional: local USB cameras on the PC

If you set:

```env
SAFETY_LOCAL_CAMERA_INDICES=0,1
SAFETY_LOCAL_INTERVAL_SEC=7.5
```

the server will also capture from those OpenCV device indices and enqueue them as `local_0`, `local_1`, etc., in addition to Pi HTTP ingest.

---

## 10. Rules YAML

Copy [`rules.example.yaml`](rules.example.yaml) to your own path and set:

```env
SAFETY_RULES_FILE=C:\path\to\my_rules.yaml
```

Keyword bumps (substring match on title + rationale) can raise severity, e.g. `fire` → `critical`.

---

## 11. Custom system prompt

Default prompt: [`prompts/system_v1.txt`](prompts/system_v1.txt).

Override with:

```env
SAFETY_SYSTEM_PROMPT_FILE=C:\path\to\custom_prompt.txt
```

Keep the **JSON-only** response contract if you want reliable parsing.

---

## 12. Optional webhook

Set `SAFETY_WEBHOOK_URL` to receive a POST (JSON body, or multipart with `image` + `payload` form field when a snapshot exists). Use for Home Assistant, n8n, or your own service.

---

## 13. Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `ollama_reachable: false` | Ollama running? Firewall? `OLLAMA_BASE_URL` correct? |
| Telegram: **Safety inference failed** / *All connection attempts failed* | The PC cannot open `OLLAMA_BASE_URL` (default `http://127.0.0.1:11434`). Start Ollama on the **same machine** as uvicorn, or set `OLLAMA_BASE_URL` to where Ollama actually listens. Test: `curl http://127.0.0.1:11434/api/tags`. |
| `model_configured: false` | `SAFETY_VLM_MODEL` / `OLLAMA_VLM_MODEL` set in `.env`? |
| Ingest 401 | `X-Safety-Token` must match `SAFETY_INGEST_SECRET`. |
| No Telegram | `TELEGRAM_*` set? `SAFETY_DRY_RUN=0`? Severity actually warning/critical? Debounce not suppressing duplicate warnings? |
| Ollama down but no Telegram (by design) | Default `SAFETY_NOTIFY_TELEGRAM_ON_OLLAMA_DOWN=0` skips Telegram for “can’t reach Ollama” errors (still logged to JSONL). Set to `1` if you want those alerts. |
| Parse / inference errors | Logs in JSONL; try a smaller `SAFETY_MAX_IMAGE_SIDE` or a different VLM; ensure the model supports images. |
| Pi: no live picture | Open **`http://127.0.0.1:9080/`** (or your `SAFETY_PI_PREVIEW_PORT`) in Chromium; ensure `pip install fastapi uvicorn`. Firewall: allow that TCP port on the Pi if viewing from another device. |
| Pi upload failures | PC IP/port, firewall on PC for 8766, `SAFETY_HTTP_TIMEOUT`, Wi‑Fi stability. |

---

## 14. Project layout (safety-related)

| Path | Role |
|------|------|
| `safety_monitor/` | FastAPI app, pipeline, Ollama client, memory, decision, notifiers |
| `safety_monitor/server.py` | HTTP routes |
| `safety_monitor/ollama_process.py` | Optional `ollama serve` on startup (local URL only) |
| `safety_monitor/main.py` | `python -m safety_monitor.main` entry |
| `safety_pi_sender.py` | Pi uploader script (repo root) |
| `requirements_safety.txt` | PC dependencies |
| `requirements_safety_pi.txt` | Pi uploader dependencies |
| `safety_monitor.env.example` | Environment template |

---

## 15. Quick start checklist

- [ ] Ollama running on PC; vision model pulled; `SAFETY_VLM_MODEL` set  
- [ ] `pip install -r requirements_safety.txt` on PC  
- [ ] `.env` from `safety_monitor.env.example` with Telegram + model  
- [ ] `python -m safety_monitor.main` — `/safety/health` OK  
- [ ] Pi: `pip install -r requirements_safety_pi.txt`  
- [ ] `python safety_pi_sender.py --server http://PC_IP:8766`  
- [ ] Optional: start with `SAFETY_DRY_RUN=1`, then disable when satisfied  
