"""Load settings from environment (.env) and optional YAML overlay."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class Settings:
    # Server
    host: str = "0.0.0.0"
    port: int = 8766
    ingest_secret: str = ""

    # Ollama
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = ""
    ollama_timeout_sec: float = 120.0
    ollama_think: bool = False
    # If True and OLLAMA_BASE_URL is localhost, run `ollama serve` when API is down
    auto_start_ollama: bool = True
    ollama_start_wait_sec: int = 90
    # Print each assistant reply (or error) to the server terminal
    print_ollama_response: bool = True
    ollama_log_max_chars: int = 8000

    # Image → model
    max_image_side: int = 896
    jpeg_quality: int = 85

    # Memory / change gate
    memory_max_entries: int = 8
    gate_width: int = 160
    gate_height: int = 90
    skip_vlm_on_static: bool = True
    static_diff_threshold: float = 4.0  # mean abs diff on 0..255 gray, scaled window
    static_frames_before_skip: int = 3

    # Notifications
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    telegram_parse_mode: str = ""
    notify_on_safe: bool = False
    notify_on_warning: bool = True
    notify_on_critical: bool = True
    # If True, send Telegram/webhook when Ollama cannot be reached (connection errors).
    # If False (default), those failures are only written to JSONL — avoids alert spam while Ollama is off.
    notify_telegram_on_ollama_unreachable: bool = False
    webhook_url: str = ""

    # Decision / debounce
    warning_debounce_sec: float = 300.0
    dry_run: bool = False

    # Paths
    system_prompt_file: str = ""
    rules_file: str = ""
    log_dir: str = "logs"
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5

    # Optional local cameras on PC (comma-separated indices, empty = none)
    local_camera_indices: List[int] = field(default_factory=list)
    local_capture_interval_sec: float = 7.5

    # Per-camera hints from YAML (camera_id -> description)
    camera_hints: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Settings":
        try:
            from dotenv import load_dotenv

            load_dotenv(_REPO_ROOT / ".env")
        except ImportError:
            pass

        yaml_path = os.getenv("SAFETY_CONFIG_YAML", "").strip()
        yaml_data: Dict[str, Any] = {}
        if yaml_path and Path(yaml_path).is_file():
            with open(yaml_path, encoding="utf-8") as f:
                yaml_data = yaml.safe_load(f) or {}

        def y(key: str, default: Any) -> Any:
            return yaml_data.get(key, default)

        indices_raw = os.getenv("SAFETY_LOCAL_CAMERA_INDICES", "").strip()
        local_indices: List[int] = []
        if indices_raw:
            for part in indices_raw.split(","):
                part = part.strip()
                if part.isdigit() or (part.startswith("-") and part[1:].isdigit()):
                    local_indices.append(int(part))
        elif isinstance(y("local_camera_indices", []), list):
            local_indices = [int(x) for x in y("local_camera_indices", [])]

        hints: Dict[str, str] = {}
        if isinstance(y("camera_hints", {}), dict):
            hints = {str(k): str(v) for k, v in y("camera_hints", {}).items()}

        default_prompt = str(_SCRIPT_DIR / "prompts" / "system_v1.txt")
        prompt_file = os.getenv("SAFETY_SYSTEM_PROMPT_FILE", "").strip() or y(
            "system_prompt_file", default_prompt
        )

        return cls(
            host=os.getenv("SAFETY_SERVER_HOST", y("host", "0.0.0.0")),
            port=_env_int("SAFETY_SERVER_PORT", int(y("port", 8766))),
            ingest_secret=os.getenv("SAFETY_INGEST_SECRET", y("ingest_secret", "")),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", y("ollama_base_url", "http://127.0.0.1:11434")
            ).rstrip("/"),
            ollama_model=os.getenv(
                "SAFETY_VLM_MODEL",
                os.getenv("OLLAMA_VLM_MODEL", y("ollama_model", "")),
            ).strip(),
            ollama_timeout_sec=_env_float(
                "OLLAMA_TIMEOUT_SEC", float(y("ollama_timeout_sec", 120.0))
            ),
            ollama_think=_env_bool("SAFETY_OLLAMA_THINK", bool(y("ollama_think", False))),
            auto_start_ollama=_env_bool(
                "SAFETY_AUTO_START_OLLAMA",
                bool(y("auto_start_ollama", True)),
            ),
            ollama_start_wait_sec=_env_int(
                "SAFETY_OLLAMA_START_WAIT_SEC",
                int(y("ollama_start_wait_sec", 90)),
            ),
            print_ollama_response=_env_bool(
                "SAFETY_PRINT_OLLAMA_RESPONSE",
                bool(y("print_ollama_response", True)),
            ),
            ollama_log_max_chars=_env_int(
                "SAFETY_OLLAMA_LOG_MAX_CHARS",
                int(y("ollama_log_max_chars", 8000)),
            ),
            max_image_side=_env_int("SAFETY_MAX_IMAGE_SIDE", int(y("max_image_side", 896))),
            jpeg_quality=_env_int("SAFETY_JPEG_QUALITY", int(y("jpeg_quality", 85))),
            memory_max_entries=_env_int(
                "SAFETY_MEMORY_MAX", int(y("memory_max_entries", 8))
            ),
            gate_width=_env_int("SAFETY_GATE_WIDTH", int(y("gate_width", 160))),
            gate_height=_env_int("SAFETY_GATE_HEIGHT", int(y("gate_height", 90))),
            skip_vlm_on_static=_env_bool(
                "SKIP_VLM_ON_STATIC", bool(y("skip_vlm_on_static", True))
            ),
            static_diff_threshold=_env_float(
                "SAFETY_STATIC_DIFF_THRESHOLD",
                float(y("static_diff_threshold", 4.0)),
            ),
            static_frames_before_skip=_env_int(
                "SAFETY_STATIC_FRAMES_K",
                int(y("static_frames_before_skip", 3)),
            ),
            telegram_bot_token=os.getenv(
                "TELEGRAM_BOT_TOKEN", y("telegram_bot_token", "")
            ),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", y("telegram_chat_id", "")),
            telegram_parse_mode=os.getenv(
                "TELEGRAM_PARSE_MODE", y("telegram_parse_mode", "")
            ),
            notify_on_safe=_env_bool(
                "NOTIFY_ON_SAFE", bool(y("notify_on_safe", False))
            ),
            notify_on_warning=_env_bool(
                "NOTIFY_ON_WARNING", bool(y("notify_on_warning", True))
            ),
            notify_on_critical=_env_bool(
                "NOTIFY_ON_CRITICAL", bool(y("notify_on_critical", True))
            ),
            notify_telegram_on_ollama_unreachable=_env_bool(
                "SAFETY_NOTIFY_TELEGRAM_ON_OLLAMA_DOWN",
                bool(y("notify_telegram_on_ollama_unreachable", False)),
            ),
            webhook_url=os.getenv("SAFETY_WEBHOOK_URL", y("webhook_url", "")),
            warning_debounce_sec=_env_float(
                "SAFETY_WARNING_DEBOUNCE_SEC",
                float(y("warning_debounce_sec", 300.0)),
            ),
            dry_run=_env_bool("SAFETY_DRY_RUN", bool(y("dry_run", False))),
            system_prompt_file=prompt_file,
            rules_file=os.getenv("SAFETY_RULES_FILE", y("rules_file", "")),
            log_dir=os.getenv("SAFETY_LOG_DIR", y("log_dir", "logs")),
            log_max_bytes=_env_int(
                "SAFETY_LOG_MAX_BYTES", int(y("log_max_bytes", 10 * 1024 * 1024))
            ),
            log_backup_count=_env_int(
                "SAFETY_LOG_BACKUP_COUNT", int(y("log_backup_count", 5))
            ),
            local_camera_indices=local_indices,
            local_capture_interval_sec=_env_float(
                "SAFETY_LOCAL_INTERVAL_SEC",
                float(y("local_capture_interval_sec", 7.5)),
            ),
            camera_hints=hints,
        )
