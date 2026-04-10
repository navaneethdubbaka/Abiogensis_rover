"""Run the safety monitor PC server: uvicorn safety_monitor.server:app"""

from __future__ import annotations

import uvicorn

from safety_monitor.config import Settings


def main() -> None:
    s = Settings.load()
    uvicorn.run(
        "safety_monitor.server:app",
        host=s.host,
        port=s.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
