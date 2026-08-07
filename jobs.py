"""Job queue and the render orchestration.

Deliberately in-process and in-memory: for a single small box this is the
right amount of machinery. If you ever outgrow one worker, swap JobStore for
Redis and this module is the only thing that changes.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from config import settings
import bgm as bgm_mod
import captions as captions_mod
import compose
import subject as subject_mod
from ffmpeg_utils import probe_duration
from providers import get_provider

log = logging.getLogger(__name__)

Status = Literal["queued", "running", "done", "error"]


@dataclass
class Job:
    id: str
    prompt: str
    style: str = "card"
    mood: str = "calm"
    bgm_volume: float = 0.22
    captions: bool = False
    motion: str = "auto"

    status: Status = "queued"
    progress: int = 0
    step: str = "Queued"
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    workdir: Path | None = None
    photo_path: Path | None = None
    voice_path: Path | None = None
    bgm_path: Path | None = None      # user-uploaded music, if any
    video_path: Path | None = None
    thumb_path: Path | None = None
    duration: float | None = None

    def public(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "step": self.step,
            "error": self.error,
            "duration": round(self.duration, 1) if self.duration else None,
            "video_url": f"/api/jobs/{self.id}/video" if self.status == "done" else None,
            "thumb_url": (
                f"/api/jobs/{self.id}/thumb"
                if self.status == "done" and self.thumb_path
                else None
            ),
        }


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._sem = asyncio.Semaphore(settings.max_concurrent_renders)

    def create(self, **kwargs) -> Job:
        job_id = uuid.uuid4().hex[:12]
        workdir = settings.data_dir / job_id
        workdir.mkdir(parents=True, exist_ok=True)
        job = Job(id=job_id, workdir=workdir, **kwargs)
        self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    async def run(self, job: Job) -> None:
        async with self._sem:
            await _process(job)

    def sweep(self) -> int:
        """Drop jobs past their TTL so disk doesn't creep up on a small host."""
        cutoff = time.time() - settings.job_ttl_minutes * 60
        removed = 0
        for job_id, job in list(self._jobs.items()):
            ref = job.finished_at or job.created_at
            if job.status in ("done", "error") and ref < cutoff:
                if job.workdir and job.workdir.exists():
                    shutil.rmtree(job.workdir, ignore_errors=True)
                del self._jobs[job_id]
                removed += 1
        return removed


store = JobStore()


# ---------------------------------------------------------------------------
def _set(job: Job, progress: int, step: str) -> None:
    job.progress = progress
    job.step = step
    log.info("[%s] %d%% %s", job.id, progress, step)


async def _process(job: Job) -> None:
    assert job.workdir and job.photo_path and job.voice_path
    wd = job.workdir
    loop = asyncio.get_running_loop()

    try:
        job.status = "running"

        # 1. How long is the video? The voice note decides.
        _set(job, 5, "Reading your voice note")
        voice_dur = await probe_duration(job.voice_path)
        if voice_dur <= 0.2:
            raise ValueError("The voice note appears to be empty or unreadable.")
        capped = min(voice_dur, settings.max_duration_sec)
        job.duration = capped + compose.TAIL_SECONDS

        # 2. Background from the prompt.
        _set(job, 15, "Generating the background")
        provider = get_provider()
        bg_path = wd / "background.png"
        await provider.generate(job.prompt, bg_path, settings.width, settings.height)

        # 3. Prepare the photo layer. Pillow is blocking - keep it off the loop.
        _set(job, 40, "Preparing your photo")
        subj_path = wd / "subject.png"
        _, _, subj_h = await loop.run_in_executor(
            None,
            lambda: subject_mod.build_subject_layer(
                job.photo_path, subj_path, style=job.style,
                # Height of the visible photo; the shadow adds ~12% on top.
                target_height=int(settings.height * 0.58),
            ),
        )

        # 4. Captions (best-effort, never fatal).
        srt: Path | None = None
        if job.captions:
            _set(job, 50, "Transcribing for captions")
            srt = await captions_mod.generate_srt(job.voice_path, wd / "captions.srt")

        # 5. Music: user upload wins, otherwise synthesize a pad.
        _set(job, 60, "Preparing the background music")
        music: Path | None = job.bgm_path
        if music is None and job.mood != "none":
            music = await bgm_mod.get_bgm(job.mood)

        # 6. Render.
        _set(job, 70, "Rendering the video")
        motion = (
            compose.pick_motion(job.prompt)
            if job.motion == "auto"
            else job.motion
        )
        out = wd / "output.mp4"
        await compose.render_video(
            workdir=wd,
            background=bg_path,
            subject=subj_path,
            voice=job.voice_path,
            bgm=music,
            out_path=out,
            duration=job.duration,
            motion=motion,
            bgm_volume=job.bgm_volume,
            subtitles=srt,
            subject_height=subj_h,
        )
        job.video_path = out

        _set(job, 95, "Creating the preview")
        job.thumb_path = await compose.make_thumbnail(out, wd / "thumb.jpg")

        job.status = "done"
        job.finished_at = time.time()
        _set(job, 100, "Done")

    except Exception as exc:  # noqa: BLE001 - surface the reason to the user
        log.exception("[%s] render failed", job.id)
        job.status = "error"
        job.error = str(exc)[:600]
        job.step = "Failed"
        job.finished_at = time.time()
