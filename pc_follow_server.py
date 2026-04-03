"""
PC-side GPU server: receives JPEG frames from the Pi, runs YOLO + follow policy, returns serial commands.

Run on the PC (with CUDA):  uvicorn pc_follow_server:app --host 0.0.0.0 --port 8765

Env: same YOLO and follow knobs as human_following.py (ROBOT_YOLO_*, MIN_CONFIDENCE, etc.).
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

from follow_policy import (
    FollowState,
    follow_config_from_env,
    follow_control_step,
    select_person_box,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def resolve_yolo_model_path() -> str:
    env = os.getenv("YOLO_MODEL_PATH")
    if env and env.strip():
        return env.strip()
    for p in (
        os.path.join(_SCRIPT_DIR, "yolo26n.pt"),
        os.path.join(_SCRIPT_DIR, "sonic_lang", "yolo26n.pt"),
    ):
        if os.path.isfile(p):
            return p
    return "yolo26n.pt"


def _parse_classes_filter():
    raw = os.getenv("ROBOT_YOLO_CLASSES", "0")
    if raw is None or not str(raw).strip():
        return None
    s = str(raw).strip().lower()
    if s == "all":
        return None
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if not parts:
        return None
    return [int(p) for p in parts]


MODEL_PATH = resolve_yolo_model_path()
ROBOT_YOLO_IMGSZ = int(os.getenv("ROBOT_YOLO_IMGSZ", "320"))
ROBOT_YOLO_DEVICE = os.getenv("ROBOT_YOLO_DEVICE", "").strip() or None
ROBOT_YOLO_HALF = _env_bool("ROBOT_YOLO_HALF", False)
ROBOT_YOLO_CLASSES = _parse_classes_filter()
USE_TRACKER = _env_bool("ROBOT_USE_TRACKER", True)
TRACKER_CFG = os.getenv("ROBOT_TRACKER", "bytetrack.yaml").strip() or "bytetrack.yaml"
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.4"))

_follow_cfg = follow_config_from_env()
_model: Optional[YOLO] = None
_tracker_failed = False
_follow_state = FollowState()
_lock = threading.Lock()


def build_yolo_kwargs() -> dict:
    kw: dict = {
        "imgsz": ROBOT_YOLO_IMGSZ,
        "verbose": False,
        "conf": MIN_CONFIDENCE,
    }
    if ROBOT_YOLO_CLASSES is not None:
        kw["classes"] = ROBOT_YOLO_CLASSES
    if ROBOT_YOLO_DEVICE is not None:
        kw["device"] = ROBOT_YOLO_DEVICE
    if ROBOT_YOLO_HALF:
        kw["half"] = True
    return kw


YOLO_KWARGS = build_yolo_kwargs()


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH)
    return _model


def run_vision(frame: np.ndarray, use_track: bool):
    global _tracker_failed
    model = get_model()
    t0 = time.perf_counter()
    if use_track and not _tracker_failed:
        try:
            results = model.track(
                frame,
                persist=True,
                tracker=TRACKER_CFG,
                **YOLO_KWARGS,
            )
        except Exception:
            _tracker_failed = True
            results = model.predict(frame, **YOLO_KWARGS)
    else:
        results = model.predict(frame, **YOLO_KWARGS)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    return results, infer_ms


app = FastAPI(title="Human follow GPU server", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "model_loaded": _model is not None,
        "tracker_failed": _tracker_failed,
    }


@app.post("/follow/reset")
def follow_reset():
    global _follow_state
    with _lock:
        _follow_state = FollowState()
    return {"ok": True, "message": "Follow state reset"}


@app.post("/follow/step")
async def follow_step(image: UploadFile = File(...)):
    global _follow_state
    t_frame = time.perf_counter()
    try:
        raw = await image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"read body: {e}") from e

    if not raw:
        raise HTTPException(status_code=400, detail="empty image")

    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="invalid image")

    fh, fw = frame.shape[0], frame.shape[1]
    decode_ms = (time.perf_counter() - t_frame) * 1000.0

    with _lock:
        _follow_state.frame_idx += 1
        results, infer_ms = run_vision(frame, USE_TRACKER and not _tracker_failed)
        best_box, best_conf, _ctr, tid_pick = select_person_box(
            results,
            get_model().names,
            _follow_cfg,
            _follow_state.prev_track_id,
            _follow_state.prev_center,
        )
        person_found = best_box is not None
        step = follow_control_step(
            _follow_state,
            _follow_cfg,
            now=time.monotonic(),
            fw=fw,
            fh=fh,
            person_found=person_found,
            best_box=best_box,
            best_conf=float(best_conf or 0.0),
            tid_pick=tid_pick,
            run_infer=True,
        )

    total_ms = (time.perf_counter() - t_frame) * 1000.0
    bbox = list(step.bbox) if step.bbox is not None else None
    return {
        "ok": True,
        "cmd": step.cmd,
        "status": step.status,
        "bbox": bbox,
        "conf": step.conf,
        "track_id": step.track_id,
        "x_dev_smooth": step.x_dev_smooth,
        "area_ratio": step.area_ratio,
        "infer_ms": round(infer_ms, 2),
        "decode_ms": round(decode_ms, 2),
        "total_ms": round(total_ms, 2),
        "frame_w": fw,
        "frame_h": fh,
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("FOLLOW_SERVER_PORT", "8765"))
    uvicorn.run(app, host=os.getenv("FOLLOW_SERVER_HOST", "0.0.0.0"), port=port)
