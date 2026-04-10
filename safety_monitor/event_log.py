"""Rotating JSONL log for safety events."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict


class RotatingJsonl:
    def __init__(self, path: str, max_bytes: int, backup_count: int) -> None:
        self.path = Path(path)
        self.max_bytes = max(1024, max_bytes)
        self.backup_count = max(1, backup_count)

    def _rotate(self) -> None:
        for i in range(self.backup_count - 1, 0, -1):
            src = self.path.with_suffix(f"{self.path.suffix}.{i}")
            dst = self.path.with_suffix(f"{self.path.suffix}.{i + 1}")
            if src.is_file():
                if dst.is_file():
                    dst.unlink()
                shutil.move(str(src), str(dst))
        if self.path.is_file():
            dst1 = self.path.with_suffix(f"{self.path.suffix}.1")
            if dst1.is_file():
                dst1.unlink()
            shutil.move(str(self.path), str(dst1))

    def write(self, obj: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        enc = line.encode("utf-8")
        if self.path.exists() and self.path.stat().st_size + len(enc) > self.max_bytes:
            self._rotate()
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)
