"""FastAPI HTTP ingest for Pi / local clients."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from safety_monitor.capture import start_local_cameras
from safety_monitor.config import Settings
from safety_monitor.pipeline import SafetyPipeline


def _check_ingest_auth(request: Request, secret: str) -> None:
    if not secret:
        return
    token = request.headers.get("x-safety-token", "")
    if token != secret:
        raise HTTPException(status_code=401, detail="invalid or missing X-Safety-Token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    pipeline = SafetyPipeline(settings)
    await pipeline.start()
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state._local_tasks: list[asyncio.Task[None]] = []
    if settings.local_camera_indices:
        app.state._local_tasks = start_local_cameras(pipeline, settings)
    yield
    for t in getattr(app.state, "_local_tasks", []):
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    await pipeline.stop()


app = FastAPI(title="Safety Monitor", lifespan=lifespan)


@app.get("/safety/health")
async def health():
    settings: Settings = app.state.settings
    base = settings.ollama_base_url.rstrip("/")
    ollama_ok = False
    ollama_err: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{base}/api/tags")
            r.raise_for_status()
            ollama_ok = True
    except Exception as e:
        ollama_err = str(e)[:200]
    return {
        "ok": True,
        "ollama_reachable": ollama_ok,
        "ollama_error": ollama_err,
        "model_configured": bool(settings.ollama_model),
        "telegram_configured": bool(settings.telegram_bot_token and settings.telegram_chat_id),
        "dry_run": settings.dry_run,
    }


@app.post("/safety/ingest")
async def ingest_frame(
    request: Request,
    image: UploadFile = File(...),
    camera_id: str = Form("pi_cam"),
):
    settings: Settings = app.state.settings
    _check_ingest_auth(request, settings.ingest_secret)
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty image")
    pipeline: SafetyPipeline = app.state.pipeline
    await pipeline.enqueue(camera_id.strip() or "pi_cam", data)
    return {"ok": True, "queued": True, "camera_id": camera_id}


@app.post("/safety/ingest/text")
async def ingest_text(
    request: Request,
    camera_id: str = Form(...),
    text: str = Form(...),
):
    settings: Settings = app.state.settings
    _check_ingest_auth(request, settings.ingest_secret)
    cid = camera_id.strip() or "pi_cam"
    pipeline: SafetyPipeline = app.state.pipeline
    pipeline.set_operator_text(cid, text)
    return {"ok": True, "camera_id": cid}
