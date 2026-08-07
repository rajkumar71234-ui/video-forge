"""Image providers - all backends in one module.

Flat layout on purpose: this project is uploaded to GitHub by drag-and-drop,
which flattens folders, so the code is written to live in a single directory.

Adding a new backend (OpenAI, Replicate, ...) means adding one class here that
subclasses ImageProvider and registering it in get_provider().
"""
from __future__ import annotations

import abc
import base64
import colorsys
import hashlib
import io
import logging
import random
from pathlib import Path

import httpx
from PIL import Image, ImageDraw, ImageFilter

from config import settings

log = logging.getLogger(__name__)


class ImageProviderError(RuntimeError):
    """Raised when a provider cannot produce an image."""


class ImageProvider(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path:
        """Generate one background image and write it to `out_path` as PNG."""
        raise NotImplementedError


class StubImageProvider(ImageProvider):
    name = "stub"

    async def generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path:
        seed = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12], 16)
        rng = random.Random(seed)

        base_hue = (seed % 360) / 360.0
        c1 = tuple(int(v * 255) for v in colorsys.hls_to_rgb(base_hue, 0.28, 0.55))
        c2 = tuple(
            int(v * 255)
            for v in colorsys.hls_to_rgb((base_hue + 0.12) % 1.0, 0.62, 0.60)
        )

        img = Image.new("RGB", (width, height), c1)
        draw = ImageDraw.Draw(img)

        # Vertical gradient.
        for y in range(height):
            t = y / max(height - 1, 1)
            # Ease so the midtones sit lower - looks less like a default gradient.
            t = t * t * (3 - 2 * t)
            draw.line(
                [(0, y), (width, y)],
                fill=tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3)),
            )

        # Soft light blobs for depth.
        overlay = Image.new("RGB", (width, height), (0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        for _ in range(7):
            r = rng.randint(width // 8, width // 3)
            cx = rng.randint(0, width)
            cy = rng.randint(0, height)
            hue = (base_hue + rng.uniform(-0.15, 0.15)) % 1.0
            col = tuple(
                int(v * 255) for v in colorsys.hls_to_rgb(hue, rng.uniform(0.45, 0.8), 0.7)
            )
            odraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=width // 12))
        img = Image.blend(img, overlay, 0.35)

        # Vignette keeps the eye on the centre where the subject sits.
        vign = Image.new("L", (width, height), 0)
        vdraw = ImageDraw.Draw(vign)
        margin = int(min(width, height) * 0.05)
        vdraw.ellipse(
            [-margin, -margin, width + margin, height + margin], fill=255
        )
        vign = vign.filter(ImageFilter.GaussianBlur(radius=min(width, height) // 8))
        dark = Image.new("RGB", (width, height), (0, 0, 0))
        img = Image.composite(img, dark, vign.point(lambda p: int(60 + p * 0.76)))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG")
        return out_path




# Steers the model away from the two things that ruin a composite background:
# a subject standing in the middle, and text baked into the image.
PROMPT_SUFFIX = (
    " Cinematic background plate, 16:9 widescreen, photorealistic, soft "
    "depth of field, balanced lighting. The centre of the frame is open and "
    "uncluttered. No people, no faces, no text, no watermarks, no logos."
)


class GeminiImageProvider(ImageProvider):
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        self.api_key = api_key or settings.gemini_api_key
        self.model = model or settings.gemini_image_model
        self.base = settings.gemini_api_base.rstrip("/")
        if not self.api_key:
            raise ImageProviderError(
                "GEMINI_API_KEY is not set. Add it to your environment, or set "
                "IMAGE_PROVIDER=stub to run without an image API."
            )

    # -- public ---------------------------------------------------------
    async def generate(self, prompt: str, out_path: Path, width: int, height: int) -> Path:
        full_prompt = prompt.strip() + PROMPT_SUFFIX
        if self.model.startswith("imagen"):
            raw = await self._call_predict(full_prompt)
        else:
            raw = await self._call_generate_content(full_prompt)

        img = Image.open(io.BytesIO(raw)).convert("RGB")
        img = _cover_resize(img, width, height)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, "PNG")
        return out_path

    # -- endpoints ------------------------------------------------------
    async def _call_generate_content(self, prompt: str) -> bytes:
        url = f"{self.base}/models/{self.model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseModalities": ["IMAGE"]},
        }
        data = await self._post(url, payload)

        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return base64.b64decode(inline["data"])

        raise ImageProviderError(
            f"Gemini returned no image data. Response keys: {list(data)}. "
            "If this persists the model name may be wrong - check "
            "GEMINI_IMAGE_MODEL."
        )

    async def _call_predict(self, prompt: str) -> bytes:
        url = f"{self.base}/models/{self.model}:predict"
        payload = {
            "instances": [{"prompt": prompt}],
            "parameters": {"sampleCount": 1, "aspectRatio": "16:9"},
        }
        data = await self._post(url, payload)
        preds = data.get("predictions") or []
        if preds and preds[0].get("bytesBase64Encoded"):
            return base64.b64decode(preds[0]["bytesBase64Encoded"])
        raise ImageProviderError(f"Imagen returned no predictions: {list(data)}")

    async def _post(self, url: str, payload: dict) -> dict:
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
        }
        # Image generation is slow; 120s is not excessive.
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, json=payload, headers=headers)

        if resp.status_code == 429:
            raise ImageProviderError(
                "Gemini rate limit / quota exceeded. Wait a minute, or set "
                "IMAGE_PROVIDER=stub to keep rendering without the API."
            )
        if resp.status_code >= 400:
            raise ImageProviderError(
                f"Gemini API error {resp.status_code}: {resp.text[:400]}"
            )
        return resp.json()


def _cover_resize(img: Image.Image, width: int, height: int) -> Image.Image:
    """Scale to fill the target box and centre-crop the overflow.

    Never letterboxes - the background must reach every edge because the Ken
    Burns move will pan across it.
    """
    src_ratio = img.width / img.height
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        new_h = height
        new_w = max(width, int(round(height * src_ratio)))
    else:
        new_w = width
        new_h = max(height, int(round(width / src_ratio)))

    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - width) // 2
    top = (new_h - height) // 2
    return img.crop((left, top, left + width, top + height))


def get_provider(name: str | None = None) -> ImageProvider:
    """Return the configured provider.

    Falls back to the offline stub rather than crashing the whole app when a
    key is missing - a background you didn't love beats a 500 error.
    """
    name = (name or settings.image_provider).lower()

    if name == "stub":
        return StubImageProvider()

    if name == "gemini":
        try:
            return GeminiImageProvider()
        except ImageProviderError as exc:
            log.warning("Gemini unavailable (%s) - falling back to stub.", exc)
            return StubImageProvider()

    raise ImageProviderError(f"Unknown IMAGE_PROVIDER: {name!r}")
