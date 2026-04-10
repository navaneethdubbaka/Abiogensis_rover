"""Extract JSON objects from model text (same strategy as vlm_ollama_follow)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Set


def extract_json_object_with_keys(text: str, required_keys: Set[str]) -> Optional[Dict[str, Any]]:
    """Parse first JSON object in text that contains all required_keys."""
    s = text.strip()
    if not s:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", s, re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and required_keys.issubset(obj.keys()):
            return obj
    except json.JSONDecodeError:
        pass
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
            if isinstance(obj, dict) and required_keys.issubset(obj.keys()):
                return obj
    return None


def normalize_severity(raw: Any) -> str:
    if not isinstance(raw, str):
        return "safe"
    v = raw.strip().lower()
    if v in ("safe", "warning", "critical"):
        return v
    return "safe"
