"""The render itself: one ffmpeg invocation that does the whole composite.

Video : still background -> Ken Burns move -> subject overlay with a slow
        float -> optional burned-in captions -> fade in/out
Audio : voice normalised to broadcast loudness -> music ducked underneath it
        by a sidechain compressor so narration always stays intelligible
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from config import settings
from ffmpeg_utils import run

log = logging.getLogger(__name__)

MOTIONS = ("zoom_in", "zoom_out", "pan_right", "pan_left")

TAIL_SECONDS = 1.2     # breathing room after the voice ends
FADE_IN = 0.6
FADE_OUT = 0.9


def pick_motion(seed_text: str) -> str:
    """Deterministic per-prompt so a re-render looks the same."""
    h = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:8], 16)
    return MOTIONS[h % len(MOTIONS)]


def _motion_exprs(motion: str, total_frames: int) -> tuple[str, str, str]:
    """Return (z, x, y) zoompan expressions.

    Everything is expressed as a function of `on` (output frame number) rather
    than accumulated from the previous frame - that avoids the drift and
    stutter zoompan is notorious for.
    """
    n = max(total_frames, 1)
    p = f"(on/{n})"  # normalised progress 0..1

    if motion == "zoom_in":
        z = f"1.001+0.16*{p}"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif motion == "zoom_out":
        z = f"1.161-0.16*{p}"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif motion == "pan_right":
        z = f"1.12+0.03*{p}"
        x, y = f"(iw-iw/zoom)*{p}", "ih/2-(ih/zoom/2)"
    else:  # pan_left
        z = f"1.12+0.03*{p}"
        x, y = f"(iw-iw/zoom)*(1-{p})", "ih/2-(ih/zoom/2)"

    return z, x, y


async def render_video(
    *,
    workdir: Path,
    background: Path,
    subject: Path | None,
    voice: Path,
    bgm: Path | None,
    out_path: Path,
    duration: float,
    motion: str = "zoom_in",
    bgm_volume: float = 0.22,
    subtitles: Path | None = None,
    subject_height: int | None = None,
) -> Path:
    W, H, FPS = settings.width, settings.height, settings.fps
    dur = round(duration, 3)
    total_frames = int(dur * FPS)

    # ---- inputs ------------------------------------------------------
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    cmd += ["-framerate", str(FPS), "-loop", "1", "-t", str(dur), "-i", str(background)]

    idx = 1
    sub_idx = None
    if subject is not None:
        cmd += ["-framerate", str(FPS), "-loop", "1", "-t", str(dur), "-i", str(subject)]
        sub_idx = idx
        idx += 1

    cmd += ["-i", str(voice)]
    voice_idx = idx
    idx += 1

    bgm_idx = None
    if bgm is not None:
        # -stream_loop -1 repeats the 32s pad for as long as the video runs.
        cmd += ["-stream_loop", "-1", "-i", str(bgm)]
        bgm_idx = idx
        idx += 1

    # ---- video graph -------------------------------------------------
    z, x, y = _motion_exprs(motion, total_frames)
    parts: list[str] = []
    parts.append(
        f"[0:v]scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,"
        f"crop={W * 2}:{H * 2},setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d=1:s={W}x{H}:fps={FPS}[bgv]"
    )

    last = "bgv"
    if sub_idx is not None:
        # The layer was already built at the right pixel size by subject.py,
        # so we pass its native height through and skip a second resample.
        sub_h = subject_height or int(H * 0.72)
        sub_h = min(sub_h, int(H * 0.92)) // 2 * 2
        parts.append(f"[{sub_idx}:v]scale=-2:{sub_h}:flags=lanczos,setsar=1[sub]")
        # Lift the photo when captions are on, so text doesn't sit over the face.
        lift = int(H * 0.07) if (subtitles and subtitles.exists()) else 0
        # eval=frame is required for the time-dependent float on y.
        parts.append(
            f"[{last}][sub]overlay=x='(W-w)/2':"
            f"y='(H-h)/2-{lift}+14*sin(2*PI*t/7)':eval=frame:format=auto[comp]"
        )
        last = "comp"

    if subtitles is not None and subtitles.exists():
        style = (
            "FontName=DejaVu Sans,FontSize=22,PrimaryColour=&H00FFFFFF,"
            "OutlineColour=&H90000000,BorderStyle=3,Outline=2,Shadow=0,"
            "Alignment=2,MarginV=60"
        )
        # Run with cwd=workdir so we can pass a bare filename and avoid the
        # filter-graph escaping nightmare that absolute paths cause here.
        parts.append(
            f"[{last}]subtitles=filename='{subtitles.name}':"
            f"force_style='{style}'[subbed]"
        )
        last = "subbed"

    parts.append(
        f"[{last}]fade=t=in:st=0:d={FADE_IN},"
        f"fade=t=out:st={max(dur - FADE_OUT, 0):.3f}:d={FADE_OUT},"
        f"format=yuv420p[vout]"
    )

    # ---- audio graph -------------------------------------------------
    afmt = "aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo"
    parts.append(
        f"[{voice_idx}:a]{afmt},"
        # EBU R128 to -16 LUFS: the standard target for spoken web video.
        "loudnorm=I=-16:TP=-1.5:LRA=11,"
        f"apad,atrim=0:{dur},asetpts=N/SR/TB[voice]"
    )

    if bgm_idx is not None:
        parts.append("[voice]asplit=2[vmain][vsc]")
        parts.append(
            f"[{bgm_idx}:a]{afmt},volume={bgm_volume},"
            f"atrim=0:{dur},asetpts=N/SR/TB,"
            f"afade=t=in:st=0:d=2,"
            f"afade=t=out:st={max(dur - 2.5, 0):.3f}:d=2.5[bgraw]"
        )
        # Music dips whenever the voice is present, recovers in the gaps.
        parts.append(
            "[bgraw][vsc]sidechaincompress="
            "threshold=0.04:ratio=8:attack=20:release=400[bgduck]"
        )
        parts.append("[vmain][bgduck]amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        parts.append("[voice]anull[aout]")

    graph = ";".join(parts)

    cmd += [
        "-filter_complex", graph,
        "-map", "[vout]", "-map", "[aout]",
        "-c:v", "libx264",
        "-preset", settings.preset,
        "-crf", str(settings.crf),
        "-profile:v", "high", "-level", "4.1",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-g", str(FPS * 2),
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart",
        "-t", str(dur),
        str(out_path),
    ]

    log.info("rendering %s (%.1fs, motion=%s)", out_path.name, dur, motion)
    await _run_in(workdir, cmd)
    return out_path


async def _run_in(workdir: Path, cmd: list[str]) -> None:
    """ffmpeg needs cwd=workdir for the relative subtitles path."""
    import asyncio

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(workdir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await asyncio.wait_for(proc.communicate(), timeout=900)
    if proc.returncode != 0:
        from ffmpeg_utils import FFmpegError

        tail = stderr.decode("utf-8", "replace").strip().splitlines()[-25:]
        raise FFmpegError("ffmpeg render failed:\n" + "\n".join(tail))


async def make_thumbnail(video: Path, out_path: Path) -> Path | None:
    """Grab a poster frame ~1.5s in for the preview card."""
    try:
        await run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", "1.5", "-i", str(video),
                "-frames:v", "1", "-vf", "scale=640:-2",
                str(out_path),
            ],
            timeout=60,
        )
        return out_path
    except Exception as exc:  # noqa: BLE001
        log.warning("thumbnail failed: %s", exc)
        return None
