from __future__ import annotations

import json
from typing import Optional

import httpx

from safety_monitor.decision import SafetyEvent
from safety_monitor.notifiers.base import Notifier


class WebhookNotifier(Notifier):
    def __init__(self, url: str) -> None:
        self.url = url.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def send(
        self,
        event: SafetyEvent,
        *,
        image_jpeg: Optional[bytes],
        caption_extra: str = "",
    ) -> None:
        if not self.enabled:
            return
        payload = {
            "camera_id": event.camera_id,
            "severity": event.severity,
            "title": event.title,
            "rationale": event.rationale,
            "categories": event.categories,
            "recommended_actions": event.recommended_actions,
            "confidence": event.confidence,
            "diff_score": event.diff_score,
            "extra": caption_extra,
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            if image_jpeg:
                files = {"image": ("snapshot.jpg", image_jpeg, "image/jpeg")}
                data = {"payload": json.dumps(payload)}
                r = await client.post(self.url, files=files, data=data)
            else:
                r = await client.post(self.url, json=payload)
            r.raise_for_status()
