"""Optional local USB/webcam capture on the PC (same process as server)."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, List, Optional

import cv2

if TYPE_CHECKING:
    from safety_monitor.config import Settings
    from safety_monitor.pipeline import SafetyPipeline


def _grab_jpeg(cap: cv2.VideoCapture, jpeg_quality: int) -> Optional[bytes]:
    ok, frame = cap.read()
    if not ok:
        return None
    ok2, buf = cv2.imencode(
        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, jpeg_quality))]
    )
    if not ok2:
        return None
    return buf.tobytes()


def start_local_cameras(
    pipeline: "SafetyPipeline", settings: "Settings"
) -> List[asyncio.Task[None]]:
    caps: List[cv2.VideoCapture] = []
    for idx in settings.local_camera_indices:
        cap = cv2.VideoCapture(idx)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        caps.append(cap)

    async def loop() -> None:
        interval = max(1.0, settings.local_capture_interval_sec)
        try:
            while True:
                for cap, idx in zip(caps, settings.local_camera_indices):
                    data = await asyncio.to_thread(
                        _grab_jpeg, cap, settings.jpeg_quality
                    )
                    if data:
                        await pipeline.enqueue(f"local_{idx}", data)
                await asyncio.sleep(interval)
        finally:
            for c in caps:
                c.release()

    return [asyncio.create_task(loop())]
