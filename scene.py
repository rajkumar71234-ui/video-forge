"""Scene mode: redraw the person INSIDE the scene they described.

The middle ground between the two other modes:

  composite  photo sits on top of a background      free, no AI
  scene      person is redrawn into the scene       free tier  <- this file
  aigen      person actually moves and speaks       paid only

It sends the uploaded photo plus an action prompt to Gemini's image model and
gets back one still image of that person doing the thing. compose.py then adds
the camera move, the voice and the music, so it plays as a video.

Nothing here animates. The person does not move - but the picture genuinely
shows them in the scene, which is what "make my photo do something" usually
means in practice.

Filename kept to 8 characters: Windows truncates longer names on drag-upload.
"""
from __future__ import annotations

import base64
import io
import logging
import mimetypes
from pathlib import Path

import httpx
from PIL import Image

from config import settings

log = logging.getLogger(__name__)


class SceneError(RuntimeError):
    """Raised when the scene image cannot be produced."""


def build_prompt(action: str) -> str:
    """Wrap the user's action into an image-edit instruction.

    Two things matter and both are easy to get wrong:
      * say "the person in the photo" explicitly, or the model invents someone
      * ask for a full scene, or it returns a head-and-shoulders crop
    """
    action = action.strip().rstrip(".")
    return (
        "Create a single photorealistic image showing the person from the "
        f"provided photo {action}. "
        "Keep their face, skin tone, hair and general appearance clearly "
        "recognisable as the same person. Show them full-body within the "
        "scene, naturally lit and correctly scaled for the environment. "
        "Wide cinematic 16:9 composition with depth. "
        "Do not show the original photo, a photo frame, a border, a collage, "
        "or any text, captions or watermarks."
    )


async def generate_scene(
    photo: Path, action: str, out_path: Path, width: int, height: int
) -> Path:
    """Return a full-frame scene image containing the person."""
    if not settings.gemini_api_key:
        raise SceneError(
            "Scene mode needs GEMINI_API_KEY. Add it in your host's environment "
            "(the free tier is enough), or use Composite mode instead."
        )

    url = (
        f"{settings.gemini_api_base.rstrip('/')}"
        f"/models/{settings.gemini_image_model}:generateContent"
    )
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": _mime(photo),
                            "data": base64.b64encode(photo.read_bytes()).decode("ascii"),
                        }
                    },
                    {"text": build_prompt(action)},
                ],
            }
        ],
        "generationConfig": {"responseModalities": ["IMAGE"]},
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

    if resp.status_code == 429:
        raise SceneError(
            "Gemini free-tier limit reached. Wait a minute and try again, or "
            "use Composite mode which has no quota."
        )
    if resp.status_code >= 400:
        raise SceneError(f"Gemini API {resp.status_code}: {resp.text[:300]}")

    raw = _first_image(resp.json())
    if raw is None:
        raise SceneError(
            "The model returned no image. Photos of recognizable public "
            "figures are rejected by Google, and some prompts are blocked by "
            "the safety filters. Try a different photo or a simpler scene."
        )

    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = _cover(img, width, height)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


# ---------------------------------------------------------------------------
def _first_image(data: dict) -> bytes | None:
    for cand in data.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    return None


def _cover(img: Image.Image, width: int, height: int) -> Image.Image:
    """Fill the frame and centre-crop the overflow - never letterbox."""
    src, dst = img.width / img.height, width / height
    if src > dst:
        new_h, new_w = height, max(width, int(round(height * src)))
    else:
        new_w, new_h = width, max(height, int(round(width / src)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left, top = (new_w - width) // 2, (new_h - height) // 2
    return img.crop((left, top, left + width, top + height))


def _mime(path: Path) -> str:
    guess, _ = mimetypes.guess_type(path.name)
    return guess if guess and guess.startswith("image/") else "image/jpeg"
