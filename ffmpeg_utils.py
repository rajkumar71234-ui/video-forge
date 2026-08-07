"""Thin async wrappers around the ffmpeg / ffprobe binaries."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger(__name__)


class FFmpegError(RuntimeError):
    pass


def require_ffmpeg() -> None:
    for binary in ("ffmpeg", "ffprobe"):
        if shutil.which(binary) is None:
            raise FFmpegError(
                f"{binary} not found on PATH. Install it "
                "(apt-get install ffmpeg) or use the provided Dockerfile."
            )


async def run(cmd: list[str], timeout: float = 900.0) -> str:
    """Run a command, raise with the tail of stderr if it fails."""
    log.debug("exec: %s", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        raise FFmpegError(f"Command timed out after {timeout}s: {cmd[0]}")

    if proc.returncode != 0:
        tail = stderr.decode("utf-8", "replace").strip().splitlines()[-25:]
        raise FFmpegError(
            f"{cmd[0]} exited {proc.returncode}:\n" + "\n".join(tail)
        )
    return stdout.decode("utf-8", "replace")


async def probe_duration(path: Path) -> float:
    """Duration of a media file in seconds."""
    out = await run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        timeout=60,
    )
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError) as exc:
        raise FFmpegError(f"Could not read duration of {path.name}: {exc}")


async def has_audio_stream(path: Path) -> bool:
    out = await run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "json", str(path),
        ],
        timeout=60,
    )
    try:
        return bool(json.loads(out).get("streams"))
    except json.JSONDecodeError:
        return False
