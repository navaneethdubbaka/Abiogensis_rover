"""Try to start `ollama serve` locally when the safety server boots (optional)."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Optional, Tuple
from urllib.parse import urlparse

import httpx

if TYPE_CHECKING:
    from safety_monitor.config import Settings


def is_local_ollama_url(base_url: str) -> bool:
    """Only auto-start when Ollama is expected on this machine."""
    try:
        u = urlparse(base_url)
    except ValueError:
        return False
    host = (u.hostname or "").lower()
    if not host:
        return False
    return host in ("127.0.0.1", "localhost", "::1")


def find_ollama_executable() -> Optional[str]:
    exe = shutil.which("ollama")
    if exe:
        return exe
    if sys.platform == "win32":
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            candidate = os.path.join(localappdata, "Programs", "Ollama", "ollama.exe")
            if os.path.isfile(candidate):
                return candidate
    return None


def spawn_ollama_serve() -> Tuple[bool, str]:
    """Start `ollama serve` detached from this process."""
    exe = find_ollama_executable()
    if not exe:
        return False, "ollama not found (install from ollama.com or add to PATH)"

    popen_kw: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if flags:
            popen_kw["creationflags"] = flags
    else:
        popen_kw["start_new_session"] = True

    try:
        subprocess.Popen([exe, "serve"], **popen_kw)
    except OSError as e:
        return False, str(e)
    return True, ""


async def tags_reachable(base: str) -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base.rstrip('/')}/api/tags")
            return r.is_success
    except Exception:
        return False


async def ensure_ollama_running(settings: "Settings") -> None:
    if not settings.auto_start_ollama:
        return
    base = settings.ollama_base_url.rstrip("/")
    if not is_local_ollama_url(base):
        print(
            f"[safety] SAFETY_AUTO_START_OLLAMA skipped (not a local URL): {base}",
            flush=True,
        )
        return

    if await tags_reachable(base):
        print("[safety] Ollama already reachable.", flush=True)
        return

    ok, err = await asyncio.to_thread(spawn_ollama_serve)
    if not ok:
        print(f"[safety] Could not start Ollama: {err}", flush=True)
        return

    print("[safety] Launched `ollama serve`; waiting for API…", flush=True)
    wait = max(5, int(settings.ollama_start_wait_sec))
    for i in range(wait):
        await asyncio.sleep(1)
        if await tags_reachable(base):
            print(f"[safety] Ollama API ready (~{i + 1}s).", flush=True)
            return
    print(
        f"[safety] Ollama did not respond on {base} within {wait}s "
        "(start Ollama manually or check install).",
        flush=True,
    )
