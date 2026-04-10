"""End-to-end processing: gate → Ollama → rules → log → notify."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Dict, Optional

import httpx

from safety_monitor.config import Settings
from safety_monitor.decision import (
    DebounceTracker,
    SafetyEvent,
    apply_rules,
    load_rules,
    model_dict_to_event,
    should_send_notification,
)
from safety_monitor.event_log import RotatingJsonl
from safety_monitor.memory import MultiCameraMemory
from safety_monitor.ollama_client import (
    decode_jpeg_to_bgr,
    load_system_prompt,
    preprocess_bgr,
    run_safety_inference,
)
from safety_monitor.notifiers import TelegramNotifier, WebhookNotifier


class SafetyPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.memory = MultiCameraMemory(
            max_entries=settings.memory_max_entries,
            gate_w=settings.gate_width,
            gate_h=settings.gate_height,
            static_threshold=settings.static_diff_threshold,
            static_frames_k=settings.static_frames_before_skip,
            skip_on_static=settings.skip_vlm_on_static,
        )
        self.system_prompt = load_system_prompt(settings.system_prompt_file)
        self.rules = load_rules(settings.rules_file) if settings.rules_file else {}
        self.debounce = DebounceTracker(settings.warning_debounce_sec)
        log_path = Path(settings.log_dir) / "safety_events.jsonl"
        self.event_log = RotatingJsonl(
            str(log_path), settings.log_max_bytes, settings.log_backup_count
        )
        self.telegram = TelegramNotifier(
            settings.telegram_bot_token,
            settings.telegram_chat_id,
            settings.telegram_parse_mode,
        )
        self.webhook = WebhookNotifier(settings.webhook_url)
        self._operator_text: Dict[str, str] = {}
        self._queue: asyncio.Queue[tuple[str, bytes]] = asyncio.Queue(maxsize=64)
        self._client: Optional[httpx.AsyncClient] = None
        self._worker_task: Optional[asyncio.Task[None]] = None

    def set_operator_text(self, camera_id: str, text: str) -> None:
        self._operator_text[camera_id] = text[:8000]

    def get_operator_text(self, camera_id: str) -> str:
        return self._operator_text.get(camera_id, "")

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.settings.ollama_timeout_sec)
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._client:
            await self._client.aclose()
            self._client = None

    async def enqueue(self, camera_id: str, jpeg_bytes: bytes) -> None:
        await self._queue.put((camera_id, jpeg_bytes))

    async def _worker_loop(self) -> None:
        assert self._client is not None
        while True:
            camera_id, jpeg_bytes = await self._queue.get()
            try:
                await self._process_one(camera_id, jpeg_bytes)
            except Exception as e:
                print(f"[safety] pipeline error cam={camera_id}: {e}")
            finally:
                self._queue.task_done()

    def _build_user_text(self, camera_id: str, diff_score: float) -> str:
        lines = [
            f"Camera ID: {camera_id}",
            f"Visual change score (mean abs diff on downscaled grayscale): {diff_score:.2f}",
        ]
        prev = self.memory.recent_summaries(camera_id, n=4)
        if prev:
            lines.append("Recent situation headlines from this camera: " + " | ".join(prev))
        hint = self.settings.camera_hints.get(camera_id, "")
        if hint:
            lines.append(f"Zone / operator hint: {hint}")
        op = self.get_operator_text(camera_id)
        if op:
            lines.append(f"Operator message: {op}")
        lines.append(
            "Respond with the single JSON object as specified in the system message."
        )
        return "\n".join(lines)

    async def _process_one(self, camera_id: str, jpeg_bytes: bytes) -> None:
        assert self._client is not None
        try:
            bgr = decode_jpeg_to_bgr(jpeg_bytes)
        except ValueError:
            return

        gray = self.memory.fingerprint(bgr)
        run_vlm, diff_score, _streak = self.memory.gate_frame(camera_id, gray)
        if not run_vlm:
            return

        try:
            model_jpeg, fw, fh = preprocess_bgr(
                bgr, self.settings.max_image_side, self.settings.jpeg_quality
            )
        except ValueError:
            return

        user_text = self._build_user_text(camera_id, diff_score)
        parsed, infer_ms, raw, err = await run_safety_inference(
            self._client,
            jpeg_bytes=model_jpeg,
            system_prompt=self.system_prompt,
            user_text=user_text,
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
            timeout_sec=self.settings.ollama_timeout_sec,
            think=self.settings.ollama_think,
        )

        if self.settings.print_ollama_response:
            chunk = (raw if raw else err) or ""
            mx = self.settings.ollama_log_max_chars
            if mx > 0 and len(chunk) > mx:
                chunk = chunk[:mx] + "…"
            print(
                f"[safety] cam={camera_id} infer={infer_ms:.0f}ms Ollama:\n{chunk}\n---",
                flush=True,
            )

        if parsed is None:
            event = SafetyEvent(
                camera_id=camera_id,
                severity="warning",
                title="Safety inference failed",
                rationale=err or "unknown",
                categories=["other"],
                recommended_actions=["Check Ollama and SAFETY_VLM_MODEL"],
                confidence=0.0,
                diff_score=diff_score,
                skipped_static=False,
                raw_assistant=raw,
                ollama_error=err,
                infer_ms=infer_ms,
            )
        else:
            event = model_dict_to_event(
                camera_id,
                parsed,
                diff_score=diff_score,
                skipped_static=False,
                raw=raw,
                err=None,
                infer_ms=infer_ms,
            )
            self.memory.push_record(camera_id, model_jpeg, diff_score, event.title)

        event = apply_rules(event, self.rules)

        log_obj = {
            "ts": time.time(),
            "camera_id": event.camera_id,
            "severity": event.severity,
            "title": event.title,
            "rationale": event.rationale,
            "categories": event.categories,
            "confidence": event.confidence,
            "diff_score": event.diff_score,
            "infer_ms": event.infer_ms,
            "ollama_error": event.ollama_error,
        }
        self.event_log.write(log_obj)

        if self.settings.dry_run:
            return

        err_text = ((event.ollama_error or "") + (event.rationale or "")).lower()
        ollama_unreachable = (
            event.title == "Safety inference failed"
            and (
                "cannot connect to ollama" in err_text
                or "connection attempts failed" in err_text
                or "connection refused" in err_text
                or "ollama request failed" in err_text
            )
        )
        if ollama_unreachable and not self.settings.notify_telegram_on_ollama_unreachable:
            return

        if not should_send_notification(
            event,
            notify_safe=self.settings.notify_on_safe,
            notify_warning=self.settings.notify_on_warning,
            notify_critical=self.settings.notify_on_critical,
        ):
            return

        allow, _reason = self.debounce.should_notify(event)
        if not allow:
            return

        caption_extra = f"conf={event.confidence:.2f} Δ={diff_score:.1f} {fw}x{fh}"
        image = None if event.severity == "safe" else model_jpeg

        if self.telegram.enabled:
            try:
                await self.telegram.send(event, image_jpeg=image, caption_extra=caption_extra)
            except Exception as e:
                print(f"[safety] telegram error: {e}")
        if self.webhook.enabled:
            try:
                await self.webhook.send(event, image_jpeg=image, caption_extra=caption_extra)
            except Exception as e:
                print(f"[safety] webhook error: {e}")
