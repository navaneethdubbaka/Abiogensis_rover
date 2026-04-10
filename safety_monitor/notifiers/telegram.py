from __future__ import annotations

from typing import Optional

import httpx

from safety_monitor.decision import SafetyEvent
from safety_monitor.notifiers.base import Notifier

_TELEGRAM = "https://api.telegram.org"


class TelegramNotifier(Notifier):
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        parse_mode: str = "",
    ) -> None:
        self.bot_token = bot_token.strip()
        self.chat_id = chat_id.strip()
        self.parse_mode = parse_mode.strip()

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)

    def _caption(self, event: SafetyEvent, extra: str) -> str:
        parts = [
            f"<b>{event.severity.upper()}</b>",
            f"cam: {event.camera_id}",
            event.title,
            (event.rationale[:800] if event.rationale else ""),
        ]
        if extra:
            parts.append(extra)
        text = "\n".join(p for p in parts if p)
        if len(text) > 1024:
            text = text[:1020] + "…"
        return text

    async def send(
        self,
        event: SafetyEvent,
        *,
        image_jpeg: Optional[bytes],
        caption_extra: str = "",
    ) -> None:
        if not self.enabled:
            return
        caption = self._caption(event, caption_extra)
        async with httpx.AsyncClient(timeout=60.0) as client:
            if image_jpeg:
                url = f"{_TELEGRAM}/bot{self.bot_token}/sendPhoto"
                data = {"chat_id": self.chat_id, "caption": caption}
                if self.parse_mode:
                    data["parse_mode"] = self.parse_mode
                files = {"photo": ("snapshot.jpg", image_jpeg, "image/jpeg")}
                r = await client.post(url, data=data, files=files)
                r.raise_for_status()
                body = r.json()
                if not body.get("ok"):
                    raise RuntimeError(f"Telegram sendPhoto: {body!r}")
            else:
                murl = f"{_TELEGRAM}/bot{self.bot_token}/sendMessage"
                payload: dict = {"chat_id": self.chat_id, "text": caption}
                if self.parse_mode:
                    payload["parse_mode"] = self.parse_mode
                r = await client.post(murl, json=payload)
                r.raise_for_status()
                body = r.json()
                if not body.get("ok"):
                    raise RuntimeError(f"Telegram sendMessage: {body!r}")
