"""
Ollama vision LLM step: JPEG -> /api/chat with base64 image -> JSON {"cmd": ...} -> validated serial command.

Used by pc_follow_server when PC_FOLLOW_MODE=vlm. Env: OLLAMA_BASE_URL, OLLAMA_VLM_MODEL, VLM_*, etc.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class VlmStepResult:
    cmd: str
    status: str
    infer_ms: float
    raw_assistant: Optional[str] = None
    ollama_error: Optional[str] = None


def _vlm_bounds() -> Tuple[int, int, int, int]:
    """min_turn, max_turn, max_forward, max_back."""
    min_turn = int(os.getenv("VLM_MIN_TURN", "115"))
    max_turn = int(os.getenv("VLM_MAX_TURN", "200"))
    max_forward = int(os.getenv("VLM_MAX_FORWARD", "200"))
    max_back = int(os.getenv("VLM_MAX_BACK", str(max_forward)))
    return min_turn, max_turn, max_forward, max_back


def _system_prompt() -> str:
    custom = os.getenv("VLM_SYSTEM_PROMPT", "").strip()
    if custom:
        # Gemma 4: do not put <|think|> here — that enables thinking (Ollama readme).
        return custom
    task = os.getenv(
        "VLM_TASK",
        "You drive a small rover from the camera view. Choose one motor command that best achieves the task.",
    ).strip()
    min_t, max_t, max_f, max_b = _vlm_bounds()
    return (
        f"Mission: {task}\n\n"
        "You must respond with exactly one JSON object, no markdown, no other text. "
        'Schema: {"cmd":"<COMMAND>","reason":"short"}\n\n'
        "Allowed COMMAND formats (Arduino serial, one line):\n"
        f"- F:n forward, n integer {min_t}..{max_f}\n"
        f"- B:n backward, n integer {min_t}..{max_b}\n"
        f"- L:n turn left, n integer {min_t}..{max_t}\n"
        f"- R:n turn right, n integer {min_t}..{max_t}\n"
        "- P:n pan/servo angle, n integer 0..180\n"
        "- S:0 stop immediately\n\n"
        "If the scene is unclear, unsafe, or you are not confident, use {\"cmd\":\"S:0\",\"reason\":\"uncertain\"}."
    )


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    s = text.strip()
    if not s:
        return None
    # Strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "cmd" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    # Prefer later `{...}` matches so Gemma-style leading "thought" text does not steal parse.
    starts = [i for i, c in enumerate(s) if c == "{"]
    for start in reversed(starts):
        for end in range(len(s), start + 2, -1):
            if s[end - 1] != "}":
                continue
            chunk = s[start:end]
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "cmd" in obj:
                return obj
    return None


def _validate_and_clamp_cmd(raw_cmd: str) -> Tuple[str, str]:
    """
    Returns (canonical_cmd, status_suffix). status_suffix empty if ok, else vlm_clamped or vlm_invalid.
    """
    raw_cmd = raw_cmd.strip()
    min_t, max_t, max_f, max_b = _vlm_bounds()

    if raw_cmd.upper() == "S:0" or re.match(r"^S:\s*0$", raw_cmd, re.IGNORECASE):
        return "S:0", ""

    m = re.match(r"^([FLRB]):(\d+)$", raw_cmd, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        n = int(m.group(2))
        if letter == "F":
            n2 = max(min_t, min(max_f, n))
            return f"F:{n2}", "" if n2 == n else "vlm_clamped"
        if letter == "B":
            n2 = max(min_t, min(max_b, n))
            return f"B:{n2}", "" if n2 == n else "vlm_clamped"
        if letter in ("L", "R"):
            n2 = max(min_t, min(max_t, n))
            return f"{letter}:{n2}", "" if n2 == n else "vlm_clamped"

    m = re.match(r"^P:(\d+)$", raw_cmd, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        n2 = max(0, min(180, n))
        return f"P:{n2}", "" if n2 == n else "vlm_clamped"

    return "S:0", "vlm_invalid"


def run_vlm_follow_step(
    jpeg_bytes: bytes,
    *,
    frame_w: int,
    frame_h: int,
    last_cmd: Optional[str] = None,
) -> VlmStepResult:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_VLM_MODEL", "").strip()
    if not model:
        return VlmStepResult(
            cmd="S:0",
            status="vlm_no_model",
            infer_ms=0.0,
            ollama_error="OLLAMA_VLM_MODEL is not set",
        )

    timeout = float(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
    include_last = _env_bool("VLM_INCLUDE_LAST_CMD", False)

    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    user_lines = [
        f"The JPEG image is from the rover camera ({frame_w}x{frame_h}).",
    ]
    if include_last and last_cmd:
        user_lines.append(f"Previous validated command was: {last_cmd}")
    user_lines.append(
        "Output only the JSON object with keys cmd and reason as specified in the system message."
    )
    user_content = "\n".join(user_lines)

    payload: Dict[str, Any] = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": user_content,
                "images": [b64],
            },
        ],
    }
    # Ollama /api/chat: thinking models (e.g. Gemma 4). Default off for latency + stable JSON tail.
    if _env_bool("VLM_OLLAMA_THINK", False):
        payload["think"] = True
    else:
        payload["think"] = False

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{base}/api/chat", json=payload)
            r.raise_for_status()
            body = r.json()
    except httpx.HTTPStatusError as e:
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return VlmStepResult(
            cmd="S:0",
            status="vlm_ollama_http_error",
            infer_ms=infer_ms,
            ollama_error=f"HTTP {e.response.status_code}: {e.response.text[:500]}",
        )
    except (httpx.RequestError, json.JSONDecodeError, KeyError) as e:
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return VlmStepResult(
            cmd="S:0",
            status="vlm_ollama_error",
            infer_ms=infer_ms,
            ollama_error=str(e)[:500],
        )

    infer_ms = (time.perf_counter() - t0) * 1000.0

    msg = body.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        return VlmStepResult(
            cmd="S:0",
            status="vlm_no_content",
            infer_ms=infer_ms,
            raw_assistant=str(content),
        )

    parsed = _extract_json_object(content)
    if not parsed or "cmd" not in parsed:
        return VlmStepResult(
            cmd="S:0",
            status="vlm_parse_error",
            infer_ms=infer_ms,
            raw_assistant=content[:2000],
        )

    raw_cmd = parsed.get("cmd")
    if not isinstance(raw_cmd, str):
        return VlmStepResult(
            cmd="S:0",
            status="vlm_parse_error",
            infer_ms=infer_ms,
            raw_assistant=content[:2000],
        )

    cmd, tag = _validate_and_clamp_cmd(raw_cmd)
    if tag == "vlm_invalid":
        status = "vlm_invalid_cmd"
    elif tag == "vlm_clamped":
        status = "vlm_ok_clamped"
    else:
        status = "vlm_ok"

    return VlmStepResult(cmd=cmd, status=status, infer_ms=infer_ms, raw_assistant=content[:500])


def ollama_reachable() -> Tuple[bool, Optional[str]]:
    base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        with httpx.Client(timeout=3.0) as client:
            r = client.get(f"{base}/api/tags")
            r.raise_for_status()
        return True, None
    except Exception as e:
        return False, str(e)[:200]
