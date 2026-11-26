"""FloraVision HTTP API: prediction, uploads, retraining, and production evaluation."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
import zipfile
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.config import AUTO_RETRAIN, CLASSES, FEATURE_COLUMNS, ROOT
from src.model import evaluate_current, last_retrain, load_story, retrain, retrain_status
from src.prediction import load_artifact, predict_bytes, predict_path
from src.preprocessing import incoming_counts, save_incoming_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("floravision")

UI_DIR = ROOT / "ui"
EXAMPLES = UI_DIR / "static" / "examples"
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_FILES = 40
MAX_ZIP_BYTES = 40 * 1024 * 1024

STARTED = time.time()
STARTED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
LATENCIES: deque[float] = deque(maxlen=400)
REQUESTS = 0
RETRAIN_LOCK = threading.Lock()

app = FastAPI(title="FloraVision", version="1.0.0")
app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@app.middleware("http")
async def record_latency(request, call_next):
    global REQUESTS
    started = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - started) * 1000
    path = request.url.path
    if path not in {"/status", "/health"} and not path.startswith("/static"):
        LATENCIES.append(elapsed_ms)
        REQUESTS += 1
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    response.headers["Cache-Control"] = "no-store"
    return response


def _artifact_or_none():
    try:
        return load_artifact()
    except Exception as exc:
        LOGGER.warning("Model is not loaded: %s", exc)
        return None


def _latency_block() -> dict:
    values = list(LATENCIES)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None}
    return {
        "count": len(values),
        "mean": round(sum(values) / len(values), 1),
        "p50": round(_percentile(values, 0.50) or 0, 1),
        "p95": round(_percentile(values, 0.95) or 0, 1),
    }


def _status_body() -> dict:
    artifact = _artifact_or_none()
    loaded = artifact is not None
    trigger = retrain_status(artifact) if loaded else {
        "pending": sum(incoming_counts().values()),
        "pending_by_class": incoming_counts(),
        "drift": None,
        "needs_retrain": False,
        "threshold": None,
        "drift_threshold": None,
        "reasons": [],
    }
    metrics = artifact.get("test_metrics") if artifact else None
    return {
        "status": "ok" if loaded else "degraded",
        "model_loaded": loaded,
        "uptime_seconds": round(time.time() - STARTED, 1),
        "started_at": STARTED_AT,
        "app": "FloraVision",
        "model_version": None if artifact is None else artifact.get("version"),
        "estimator": None if artifact is None else artifact.get("estimator_name"),
        "trained_at": None if artifact is None else artifact.get("trained_at"),
        "classes": CLASSES,
        "feature_count": len(FEATURE_COLUMNS),
        "test_metrics": metrics,
        "requests_served": REQUESTS,
        "latency_ms": _latency_block(),
        "retrain": trigger,
        "last_retrain": last_retrain(),
        "auto_retrain": AUTO_RETRAIN,
        "retrain_running": RETRAIN_LOCK.locked(),
    }


def _start_retrain(force: bool) -> bool:
    if not RETRAIN_LOCK.acquire(blocking=False):
        return False

    def job() -> None:
        try:
            result = retrain(force=force)
            LOGGER.info("Retrain finished: %s", result.get("message"))
        except Exception:
            LOGGER.exception("Retrain failed")
        finally:
            RETRAIN_LOCK.release()

    threading.Thread(target=job, daemon=True).start()
    return True


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health():
    artifact = _artifact_or_none()
    body = {
        "status": "ok" if artifact else "degraded",
        "model_loaded": artifact is not None,
        "uptime_seconds": round(time.time() - STARTED, 1),
    }
    if artifact is None:
        return JSONResponse(status_code=503, content=body)
    return body


@app.get("/status")
def status() -> dict:
    return _status_body()


@app.get("/story")
def story() -> dict:
    return load_story()


@app.get("/metrics")
def metrics() -> dict:
    artifact = _artifact_or_none()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    return {
        "version": artifact.get("version"),
        "estimator": artifact.get("estimator_name"),
        "trained_at": artifact.get("trained_at"),
        "cv": artifact.get("cv"),
        "test": artifact.get("test_metrics"),
        "comparison": artifact.get("comparison"),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The upload was empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=400, detail="Image is larger than 8 MB.")
    try:
        return predict_bytes(raw)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail="That file could not be read as an image.") from exc


@app.post("/predict/example/{label}")
def predict_example(label: str) -> dict:
    if label not in CLASSES:
        raise HTTPException(status_code=404, detail="Unknown species.")
    path = EXAMPLES / f"{label}.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Example image is not available yet.")
    try:
        return predict_path(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/upload")
async def upload(label: str = Form(...), files: list[UploadFile] = File(...)) -> dict:
    if label not in CLASSES:
        raise HTTPException(status_code=400, detail=f"Label must be one of {CLASSES}.")
    if len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Upload at most {MAX_FILES} images at a time.")
    saved = []
    rejected = []
    for item in files:
        raw = await item.read()
        if len(raw) > MAX_IMAGE_BYTES:
            rejected.append({"file": item.filename, "error": "larger than 8 MB"})
            continue
        try:
            path = save_incoming_image(raw, label)
            saved.append(path.name)
        except ValueError as exc:
            rejected.append({"file": item.filename, "error": str(exc)})
    return _upload_result(saved, rejected)


@app.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)) -> dict:
    raw = await file.read()
    if len(raw) > MAX_ZIP_BYTES:
        raise HTTPException(status_code=400, detail="Zip is larger than 40 MB.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="That file is not a zip archive.") from exc

    saved: list[str] = []
    rejected: list[dict] = []
    with archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            parts = [part for part in name.split("/") if part and part != "."]
            if not parts or any(part == ".." for part in parts):
                continue
            if len(parts) < 2:
                rejected.append({"file": name, "error": "put images inside a species folder"})
                continue
            label = parts[-2].lower()
            if label not in CLASSES:
                rejected.append({"file": name, "error": f"folder {label!r} is not a species"})
                continue
            if len(saved) >= 200:
                rejected.append({"file": name, "error": "stopped after 200 images"})
                continue
            payload = archive.read(info)
            if len(payload) > MAX_IMAGE_BYTES:
                rejected.append({"file": name, "error": "larger than 8 MB"})
                continue
            try:
                path = save_incoming_image(payload, label)
                saved.append(f"{label}/{path.name}")
            except ValueError as exc:
                rejected.append({"file": name, "error": str(exc)})
    return _upload_result(saved, rejected)


def _upload_result(saved: list[str], rejected: list[dict]) -> dict:
    artifact = _artifact_or_none()
    trigger = retrain_status(artifact)
    started = False
    if AUTO_RETRAIN and trigger["needs_retrain"] and saved:
        started = _start_retrain(False)
    return {
        "saved": len(saved),
        "files": saved,
        "rejected": rejected,
        "retrain": trigger,
        "retrain_started": started,
    }


@app.post("/retrain")
async def retrain_now(force: bool = False) -> dict:
    if _artifact_or_none() is None:
        raise HTTPException(status_code=503, detail="Train a model before retraining.")
    if not RETRAIN_LOCK.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="A retrain is already running.")
    try:
        result = await asyncio.to_thread(retrain, force)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        RETRAIN_LOCK.release()
    return result


@app.post("/evaluate")
def evaluate() -> dict:
    """Score the model that is loaded right now against the held-out test photographs."""
    artifact = _artifact_or_none()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    try:
        metrics = evaluate_current(artifact["pipeline"])
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "model_version": artifact.get("version"),
        "estimator": artifact.get("estimator_name"),
        "trained_at": artifact.get("trained_at"),
        "metrics": metrics,
        "note": "Scored in this running service on data/test. The test images were not used to choose the model.",
    }
