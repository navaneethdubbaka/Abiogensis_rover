"""Normalize model output, YAML rules, debounce, notify gating."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from safety_monitor.json_utils import normalize_severity


@dataclass
class SafetyEvent:
    camera_id: str
    severity: str
    title: str
    rationale: str
    categories: List[str]
    recommended_actions: List[str]
    confidence: float
    diff_score: float
    skipped_static: bool
    raw_assistant: Optional[str] = None
    ollama_error: Optional[str] = None
    infer_ms: float = 0.0


def _as_list(v: Any) -> List[str]:
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v]
    return [str(v)]


def model_dict_to_event(
    camera_id: str,
    d: Dict[str, Any],
    *,
    diff_score: float,
    skipped_static: bool,
    raw: Optional[str] = None,
    err: Optional[str] = None,
    infer_ms: float = 0.0,
) -> SafetyEvent:
    conf = d.get("confidence", 0.5)
    try:
        confidence = float(conf)
    except (TypeError, ValueError):
        confidence = 0.0
    return SafetyEvent(
        camera_id=camera_id,
        severity=normalize_severity(d.get("severity")),
        title=str(d.get("title", ""))[:500],
        rationale=str(d.get("rationale", ""))[:4000],
        categories=_as_list(d.get("categories")),
        recommended_actions=_as_list(d.get("recommended_actions")),
        confidence=max(0.0, min(1.0, confidence)),
        diff_score=diff_score,
        skipped_static=skipped_static,
        raw_assistant=raw,
        ollama_error=err,
        infer_ms=infer_ms,
    )


def load_rules(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def apply_rules(event: SafetyEvent, rules: Dict[str, Any]) -> SafetyEvent:
    """Keyword bumps from rules file."""
    bumps = rules.get("severity_bump_keywords") or {}
    if not isinstance(bumps, dict):
        return event
    text = (event.title + " " + event.rationale).lower()
    sev = event.severity
    for keyword, target in bumps.items():
        if str(keyword).lower() in text:
            tgt = str(target).lower()
            if tgt == "critical":
                sev = "critical"
            elif tgt == "warning" and sev == "safe":
                sev = "warning"
    if sev != event.severity:
        return SafetyEvent(
            camera_id=event.camera_id,
            severity=sev,
            title=event.title,
            rationale=event.rationale,
            categories=event.categories,
            recommended_actions=event.recommended_actions,
            confidence=event.confidence,
            diff_score=event.diff_score,
            skipped_static=event.skipped_static,
            raw_assistant=event.raw_assistant,
            ollama_error=event.ollama_error,
            infer_ms=event.infer_ms,
        )
    return event


@dataclass
class DebounceState:
    last_key: str = ""
    last_time: float = 0.0


class DebounceTracker:
    def __init__(self, warning_sec: float) -> None:
        self.warning_sec = warning_sec
        self._by_cam: Dict[str, DebounceState] = {}

    def _sig(self, event: SafetyEvent) -> str:
        h = hashlib.sha256(
            f"{event.severity}|{event.title}|{event.rationale[:200]}".encode()
        ).hexdigest()[:16]
        return h

    def should_notify(self, event: SafetyEvent) -> Tuple[bool, str]:
        if event.severity == "critical":
            return True, "critical"
        st = self._by_cam.setdefault(event.camera_id, DebounceState())
        sig = self._sig(event)
        now = time.time()
        if sig == st.last_key and (now - st.last_time) < self.warning_sec:
            return False, f"debounced_{event.severity}"
        st.last_key = sig
        st.last_time = now
        return True, event.severity


def should_send_notification(
    event: SafetyEvent,
    *,
    notify_safe: bool,
    notify_warning: bool,
    notify_critical: bool,
) -> bool:
    if event.severity == "critical" and notify_critical:
        return True
    if event.severity == "warning" and notify_warning:
        return True
    if event.severity == "safe" and notify_safe:
        return True
    return False
