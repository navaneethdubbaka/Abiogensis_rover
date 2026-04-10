"""Async Ollama /api/chat for safety JSON output."""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import httpx
import numpy as np

from safety_monitor.json_utils import extract_json_object_with_keys

_REQUIRED = {"severity", "title", "rationale"}


def load_system_prompt(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return (
        "You are a safety monitoring assistant. Respond with one JSON object with keys: "
        'severity (safe|warning|critical), categories (array), title, rationale, '
        'recommended_actions (array), confidence (number). No markdown.'
    )


def preprocess_bgr(
    bgr: np.ndarray, max_side: int, jpeg_quality: int
) -> Tuple[bytes, int, int]:
    h, w = bgr.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    if scale < 1.0:
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        bgr = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    h2, w2 = bgr.shape[:2]
    ok, buf = cv2.imencode(
        ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, jpeg_quality))]
    )
    if not ok:
        raise ValueError("jpeg encode failed")
    return buf.tobytes(), w2, h2


def decode_jpeg_to_bgr(jpeg_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("invalid jpeg")
    return img


async def run_safety_inference(
    client: httpx.AsyncClient,
    *,
    jpeg_bytes: bytes,
    system_prompt: str,
    user_text: str,
    base_url: str,
    model: str,
    timeout_sec: float,
    think: bool,
) -> Tuple[Optional[Dict[str, Any]], float, Optional[str], Optional[str]]:
    """
    Returns (parsed_json_or_none, infer_ms, raw_assistant_text, error_message).
    """
    if not model:
        return None, 0.0, None, "SAFETY_VLM_MODEL / OLLAMA_VLM_MODEL is not set"

    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "think": think,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_text,
                "images": [b64],
            },
        ],
    }
    t0 = time.perf_counter()
    try:
        r = await client.post(f"{base_url}/api/chat", json=payload)
        r.raise_for_status()
        body = r.json()
    except httpx.HTTPStatusError as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return None, ms, None, f"HTTP {e.response.status_code}: {e.response.text[:500]}"
    except (httpx.RequestError, json.JSONDecodeError, KeyError) as e:
        ms = (time.perf_counter() - t0) * 1000.0
        return None, ms, None, str(e)[:500]

    ms = (time.perf_counter() - t0) * 1000.0
    msg = body.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        return None, ms, str(content), "no string content from Ollama"

    parsed = extract_json_object_with_keys(content, _REQUIRED)
    if not parsed:
        return None, ms, content, "parse_error"
    return parsed, ms, content, None
