"""Optional burned-in captions, transcribed from the voice note by Gemini.

Caveat worth knowing: Gemini's word-level timestamps are approximate. Captions
land close enough to feel synced for narration, but this is not a substitute
for a dedicated ASR model if you need frame-accurate subtitles. If anything
here fails the render continues without captions - captions are never allowed
to break a video.
"""
from __future__ import annotations

import base64
import logging
import re
from pathlib import Path

import httpx

from config import settings
from ffmpeg_utils import run

log = logging.getLogger(__name__)

INSTRUCTION = """Transcribe this audio into SRT subtitle format.

Rules:
- Output ONLY valid SRT. No markdown fences, no commentary.
- Each cue is at most 8 words, so it fits on one line on screen.
- Timestamps must be HH:MM:SS,mmm and must not overlap or go backwards.
- Transcribe in the language actually spoken. Do not translate.
- If the audio has no intelligible speech, output nothing at all.
"""

SRT_BLOCK = re.compile(
    r"\d+\s*\n\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}"
)


async def generate_srt(voice_path: Path, out_path: Path) -> Path | None:
    if not settings.enable_captions or not settings.gemini_api_key:
        return None

    try:
        compact = out_path.parent / "voice_for_asr.mp3"
        # 16 kHz mono keeps the request small; ASR gains nothing from stereo.
        await run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(voice_path),
                "-ac", "1", "-ar", "16000", "-b:a", "48k",
                str(compact),
            ],
            timeout=180,
        )

        audio_b64 = base64.b64encode(compact.read_bytes()).decode("ascii")
        url = f"{settings.gemini_api_base.rstrip('/')}/models/{settings.gemini_text_model}:generateContent"
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inline_data": {"mime_type": "audio/mpeg", "data": audio_b64}},
                        {"text": INSTRUCTION},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0},
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                url,
                json=payload,
                headers={
                    "x-goog-api-key": settings.gemini_api_key,
                    "Content-Type": "application/json",
                },
            )
        if resp.status_code >= 400:
            log.warning("caption request failed %s: %s", resp.status_code, resp.text[:200])
            return None

        text = _extract_text(resp.json())
        srt = _clean(text)
        if not srt or not SRT_BLOCK.search(srt):
            log.info("no usable SRT returned - skipping captions.")
            return None

        out_path.write_text(srt, encoding="utf-8")
        return out_path

    except Exception as exc:  # noqa: BLE001 - captions are best-effort
        log.warning("caption generation skipped: %s", exc)
        return None


def _extract_text(data: dict) -> str:
    chunks: list[str] = []
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            if "text" in part:
                chunks.append(part["text"])
    return "\n".join(chunks)


def _clean(text: str) -> str:
    text = text.strip()
    # Models sometimes wrap output in a fence despite being told not to.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text).strip()
    return text
