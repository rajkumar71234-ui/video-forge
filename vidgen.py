"""AI video generation via Gemini Omni Flash (interactions API).

This is the "the person actually does the thing" path, as opposed to compose.py
which only moves a still photo over a background.

    reference photo + prompt  ->  Omni Flash  ->  real generated footage w/ audio

Two things the API cannot do today, straight from Google's docs:
  * "Uploading audio references is unsupported" - so the spoken dialogue comes
    out in a model-chosen voice, NOT the user's cloned voice. Cloning needs a
    separate service (ElevenLabs) plus a lip-sync pass.
  * "Uploading and editing images containing certain recognizable people is not
    supported" - photos of celebrities are rejected by Google, not by us.

Filenames in this project stay <= 8 characters because the repo is uploaded by
drag-and-drop from Windows, which silently truncates longer names to 8.3 form.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path

import httpx

from config import settings

log = logging.getLogger(__name__)


class VideoGenError(RuntimeError):
    """Raised when generation fails or is refused."""


# Steers Omni toward one continuous, usable clip. Without this the model
# invents its own multi-shot narrative, which wrecks character consistency.
STYLE_SUFFIX = (
    " Photorealistic, cinematic lighting, natural motion and physics. "
    "Keep the person's face, build and clothing consistent throughout. "
    "No on-screen text, no captions, no watermarks, no split screens."
)


def build_prompt(action: str, dialogue: str = "", seconds: int = 8) -> str:
    """Turn a plain description into a timecoded Omni prompt.

    Omni accepts natural-language timing ("After 3 seconds...") and an explicit
    timecode syntax. The timecode form gives noticeably steadier pacing.
    """
    action = action.strip().rstrip(".")
    ref = "<IMAGE_REF_0>"

    if dialogue.strip():
        d = dialogue.strip().strip('"')
        beat = max(3, seconds - 3)
        body = (
            f"[0-{beat}s] {ref} {action}. Single continuous shot.\n"
            f"[{beat}-{seconds}s] {ref} looks toward the camera and says clearly, "
            f'in sync with their lips: "{d}"\n'
            "The only speech in the video is that line. No other dialogue, "
            "no narrator."
        )
    else:
        body = (
            f"[0-{seconds}s] {ref} {action}. "
            "One single continuous unbroken shot, no scene cuts. No dialogue."
        )

    return (
        body
        + "\n\nUse the given image as a reference for the person's appearance. "
        "Do not use it as a literal first frame."
        + STYLE_SUFFIX
    )


class OmniVideoProvider:
    """Thin REST client for the Gemini Omni Flash interactions endpoint."""

    name = "omni"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.omni_model
        self.base = settings.gemini_api_base.rstrip("/")
        if not self.api_key:
            raise VideoGenError(
                "AI video mode needs GEMINI_API_KEY. Set it in your host's "
                "environment, or use Composite mode which is free."
            )

    async def generate(
        self,
        *,
        photo: Path,
        prompt: str,
        out_path: Path,
        aspect_ratio: str = "16:9",
    ) -> Path:
        payload = {
            "model": self.model,
            "input": [
                {
                    "type": "image",
                    "data": base64.b64encode(photo.read_bytes()).decode("ascii"),
                    "mime_type": _mime(photo),
                },
                {"type": "text", "text": prompt},
            ],
            "generation_config": {"video_config": {"task": "reference_to_video"}},
            # delivery=uri is required above 4 MB, and generated clips routinely
            # exceed that, so always take the URI path.
            "response_format": {
                "type": "video",
                "aspect_ratio": aspect_ratio,
                "delivery": "uri",
            },
        }

        data = await self._post(f"{self.base}/interactions", payload)
        inline, uri = _find_video(data)

        if inline:
            out_path.write_bytes(base64.b64decode(inline))
            return out_path
        if uri:
            await self._download(uri, out_path)
            return out_path

        raise VideoGenError(
            "The model returned no video. This usually means the prompt or the "
            "photo was blocked by Google's safety filters - photos of "
            f"recognizable public figures are rejected. Status: {data.get('status')}"
        )

    # -- http -----------------------------------------------------------
    async def _post(self, url: str, payload: dict) -> dict:
        headers = {"x-goog-api-key": self.api_key, "Content-Type": "application/json"}
        # Generation is slow; the API blocks until the clip is ready.
        timeout = httpx.Timeout(settings.vidgen_timeout_sec, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code == 429:
            raise VideoGenError(
                "Gemini rate limit or quota exceeded. Video generation is a paid "
                "feature - check that billing is enabled on your API key."
            )
        if resp.status_code in (400, 403) and "billing" in resp.text.lower():
            raise VideoGenError(
                "Google rejected the request for billing reasons. Video "
                "generation is not available on the free API tier."
            )
        if resp.status_code >= 400:
            raise VideoGenError(f"Gemini API {resp.status_code}: {resp.text[:400]}")
        return resp.json()

    async def _download(self, uri: str, out_path: Path) -> None:
        """Poll the Files API until ACTIVE, then download."""
        file_id = uri.rstrip("/").split("/")[-1].split(":")[0]
        headers = {"x-goog-api-key": self.api_key}
        deadline = settings.vidgen_timeout_sec

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            waited = 0.0
            while waited < deadline:
                info = await client.get(
                    f"{self.base}/files/{file_id}", headers=headers
                )
                state = ""
                if info.status_code < 400:
                    st = info.json().get("state")
                    state = st.get("name") if isinstance(st, dict) else (st or "")
                if state == "ACTIVE":
                    break
                if state == "FAILED":
                    raise VideoGenError("Google reported the generation FAILED.")
                await asyncio.sleep(5)
                waited += 5
            else:
                raise VideoGenError(
                    f"Video was not ready within {deadline:.0f}s. It may still "
                    "be processing - try again in a minute."
                )

            dl = await client.get(
                f"{self.base}/files/{file_id}:download",
                params={"alt": "media"},
                headers=headers,
                follow_redirects=True,
            )
            if dl.status_code >= 400:
                raise VideoGenError(f"Download failed {dl.status_code}: {dl.text[:200]}")
            out_path.write_bytes(dl.content)


# ---------------------------------------------------------------------------
def _find_video(data: dict) -> tuple[str | None, str | None]:
    """Pull (base64, uri) out of the interactions response.

    The SDK exposes a convenience `output_video` field but the REST API does
    not, so walk `steps[].content[]` and accept either shape.
    """
    vid = data.get("output_video")
    if isinstance(vid, dict):
        return vid.get("data"), vid.get("uri")

    for step in data.get("steps", []):
        for part in step.get("content", []) or []:
            if part.get("type") == "video" or str(
                part.get("mime_type", "")
            ).startswith("video/"):
                return part.get("data"), part.get("uri")
    return None, None


def _mime(path: Path) -> str:
    guess, _ = mimetypes.guess_type(path.name)
    if guess and guess.startswith("image/"):
        return guess
    return "image/jpeg"
