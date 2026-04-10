from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from safety_monitor.decision import SafetyEvent


class Notifier(ABC):
    @abstractmethod
    async def send(
        self,
        event: "SafetyEvent",
        *,
        image_jpeg: Optional[bytes],
        caption_extra: str = "",
    ) -> None:
        pass
