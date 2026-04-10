"""Per-camera ring buffer, visual fingerprint, and static-scene gate."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameRecord:
    ts: float
    diff_score: float
    summary: str  # last model title or ""
    jpeg_bytes: bytes  # analysis-sized jpeg for Telegram / context


@dataclass
class CameraMemoryState:
    records: Deque[FrameRecord] = field(default_factory=deque)
    prev_gray: Optional[np.ndarray] = None
    static_streak: int = 0


class MultiCameraMemory:
    def __init__(
        self,
        max_entries: int,
        gate_w: int,
        gate_h: int,
        static_threshold: float,
        static_frames_k: int,
        skip_on_static: bool,
    ) -> None:
        self.max_entries = max_entries
        self.gate_w = gate_w
        self.gate_h = gate_h
        self.static_threshold = static_threshold
        self.static_frames_k = static_frames_k
        self.skip_on_static = skip_on_static
        self._by_cam: Dict[str, CameraMemoryState] = {}

    def _state(self, camera_id: str) -> CameraMemoryState:
        if camera_id not in self._by_cam:
            self._by_cam[camera_id] = CameraMemoryState()
        return self._by_cam[camera_id]

    def fingerprint(self, bgr: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        return cv2.resize(g, (self.gate_w, self.gate_h), interpolation=cv2.INTER_AREA)

    def gate_frame(
        self, camera_id: str, gray_small: np.ndarray
    ) -> Tuple[bool, float, int]:
        """
        Returns (run_vlm, diff_score, static_streak).
        run_vlm False when scene considered static and skip_on_static and streak >= K.
        """
        st = self._state(camera_id)
        diff = 255.0
        if st.prev_gray is not None and st.prev_gray.shape == gray_small.shape:
            a = gray_small.astype(np.float32)
            b = st.prev_gray.astype(np.float32)
            diff = float(np.mean(np.abs(a - b)))
        st.prev_gray = gray_small.copy()

        if diff < self.static_threshold:
            st.static_streak += 1
        else:
            st.static_streak = 0

        if (
            self.skip_on_static
            and st.static_streak >= self.static_frames_k
            and st.prev_gray is not None
        ):
            return False, diff, st.static_streak
        return True, diff, st.static_streak

    def push_record(
        self,
        camera_id: str,
        jpeg_bytes: bytes,
        diff_score: float,
        summary: str,
    ) -> None:
        st = self._state(camera_id)
        rec = FrameRecord(ts=time.time(), diff_score=diff_score, summary=summary, jpeg_bytes=jpeg_bytes)
        st.records.append(rec)
        while len(st.records) > self.max_entries:
            st.records.popleft()

    def recent_summaries(self, camera_id: str, n: int = 3) -> List[str]:
        st = self._state(camera_id)
        out: List[str] = []
        for r in list(st.records)[-n:]:
            if r.summary:
                out.append(r.summary)
        return out
