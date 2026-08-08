"""FastAPI application: upload -> queue -> poll -> download."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from jobs import store
from bgm import MOODS
from ffmpeg_utils import require_ffmpeg
from subject import STYLES

# force=True matters: uvicorn installs its own root handlers before importing
# this module, and a plain basicConfig() is a no-op once handlers exist - which
# is why progress lines never reached the host's log viewer.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
log = logging.getLogger("videoforge")

INDEX_HTML = Path(__file__).resolve().parent / "ui.htm"

ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".bmp"}
ALLOWED_AUDIO = {".mp3", ".m4a", ".aac", ".wav", ".ogg", ".opus", ".oga", ".webm", ".flac"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    require_ffmpeg()
    log.info(
        "Video Forge up. provider=%s captions=%s cutout=%s",
        settings.image_provider, settings.enable_captions, settings.enable_cutout,
    )
    sweeper = asyncio.create_task(_sweep_loop())
    try:
        yield
    finally:
        sweeper.cancel()


async def _sweep_loop() -> None:
    while True:
        await asyncio.sleep(600)
        try:
            n = store.sweep()
            if n:
                log.info("swept %d expired job(s)", n)
        except Exception:  # noqa: BLE001
            log.exception("sweep failed")


app = FastAPI(title="Video Forge", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "provider": settings.image_provider,
        "captions_enabled": settings.enable_captions,
        "cutout_enabled": settings.enable_cutout,
        "max_duration_sec": settings.max_duration_sec,
        "max_upload_mb": settings.max_upload_mb,
        "styles": list(STYLES),
        "moods": list(MOODS) + ["none"],
        "vidgen_enabled": settings.enable_vidgen,
        "vidgen_seconds": settings.vidgen_seconds,
        # Scene mode only needs a key, not billing.
        "scene_enabled": bool(settings.gemini_api_key),
    }


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    prompt: str = Form(..., min_length=3, max_length=1200),
    photo: UploadFile = File(...),
    voice: UploadFile | None = File(None),
    music: UploadFile | None = File(None),
    mode: str = Form("composite"),
    dialogue: str = Form(""),
    style: str = Form("card"),
    mood: str = Form("calm"),
    motion: str = Form("auto"),
    bgm_volume: float = Form(0.22),
    captions: bool = Form(False),
) -> JSONResponse:
    if mode not in ("composite", "scene", "aigen"):
        raise HTTPException(400, "mode must be 'composite', 'scene' or 'aigen'")
    if mode == "scene" and not settings.gemini_api_key:
        raise HTTPException(
            400,
            "Scene mode needs GEMINI_API_KEY set on the server. The free tier "
            "is enough. Use Composite mode meanwhile - it needs no key.",
        )
    if mode == "aigen" and not settings.enable_vidgen:
        raise HTTPException(
            400,
            "AI video mode is not enabled on this server. Set ENABLE_VIDGEN=true "
            "and enable billing on the Gemini API key.",
        )
    if style not in STYLES:
        raise HTTPException(400, f"style must be one of {list(STYLES)}")
    if mood not in MOODS and mood != "none":
        raise HTTPException(400, f"mood must be one of {list(MOODS) + ['none']}")
    # Composite and Scene are both driven by the voice note's length.
    if mode in ("composite", "scene") and (voice is None or not voice.filename):
        raise HTTPException(400, f"{mode.title()} mode needs a voice note.")

    job = store.create(
        prompt=prompt.strip(),
        mode=mode,
        dialogue=dialogue.strip()[:400],
        style=style,
        mood=mood,
        motion=motion,
        bgm_volume=max(0.0, min(float(bgm_volume), 1.0)),
        captions=bool(captions) and settings.enable_captions,
    )
    wd = job.workdir
    assert wd is not None

    try:
        job.photo_path = await _save(photo, wd, "photo", ALLOWED_IMAGE)
        if voice is not None and voice.filename:
            job.voice_path = await _save(voice, wd, "voice", ALLOWED_AUDIO)
        if music is not None and music.filename:
            job.bgm_path = await _save(music, wd, "music", ALLOWED_AUDIO)
    except HTTPException:
        store.sweep()
        raise

    background_tasks.add_task(store.run, job)
    return JSONResponse(job.public(), status_code=202)


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found or expired.")
    return job.public()


@app.get("/api/jobs/{job_id}/video")
async def job_video(job_id: str):
    job = store.get(job_id)
    if job is None or job.status != "done" or not job.video_path:
        raise HTTPException(404, "Video not ready.")
    return FileResponse(
        job.video_path,
        media_type="video/mp4",
        filename=f"videoforge-{job_id}.mp4",
    )


@app.get("/api/jobs/{job_id}/thumb")
async def job_thumb(job_id: str):
    job = store.get(job_id)
    if job is None or not job.thumb_path or not job.thumb_path.exists():
        raise HTTPException(404, "No thumbnail.")
    return FileResponse(job.thumb_path, media_type="image/jpeg")


@app.delete("/api/jobs/{job_id}")
async def job_delete(job_id: str) -> dict:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(404, "Job not found.")
    job.finished_at = 0.0  # force it past the TTL
    job.status = "done" if job.status == "running" else job.status
    store.sweep()
    return {"deleted": True}


# ---------------------------------------------------------------------------
async def _save(
    upload: UploadFile, workdir: Path, stem: str, allowed: set[str]
) -> Path:
    ext = Path(upload.filename or "").suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            400,
            f"'{upload.filename}' has an unsupported type. Allowed: "
            + ", ".join(sorted(allowed)),
        )

    dest = workdir / f"{stem}{ext}"
    limit = settings.max_upload_mb * 1024 * 1024
    written = 0
    with dest.open("wb") as fh:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    413, f"'{upload.filename}' is larger than {settings.max_upload_mb} MB."
                )
            fh.write(chunk)
    if written == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"'{upload.filename}' is empty.")
    return dest


@app.get("/", include_in_schema=False)
async def index():
    """Serve the single-page UI.

    Deliberately NOT a StaticFiles mount on "/": in this flat layout the app's
    own .py source sits in the same directory, and mounting it would publish
    the source (and any .env beside it) to the internet.
    """
    return FileResponse(INDEX_HTML, media_type="text/html")
