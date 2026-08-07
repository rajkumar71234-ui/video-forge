"""Turn the uploaded photo into an RGBA layer ready to composite.

Three styles:
  cutout  - AI background removal (needs rembg), subject floats free with a shadow
  card    - portrait crop in a rounded white frame, drop shadow  [default]
  circle  - circular crop with a soft ring, good for a single face

`card` and `circle` are pure Pillow, so they always work. `cutout` degrades
gracefully to `card` when rembg is not installed.
"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from config import settings

log = logging.getLogger(__name__)

STYLES = ("card", "circle", "cutout")


def build_subject_layer(
    photo_path: Path,
    out_path: Path,
    style: str = "card",
    target_height: int = 700,
) -> tuple[Path, int, int]:
    """Build the layer and return (path, width, height).

    `target_height` is the height of the *visible* photo. The returned layer is
    taller because of the drop shadow, which is why the real size is returned -
    the compositor uses it verbatim so the photo is never resampled twice.
    """
    img = Image.open(photo_path)
    img = ImageOps.exif_transpose(img)  # honour phone camera rotation
    img = img.convert("RGBA")

    style = style if style in STYLES else "card"
    if style == "cutout":
        layer = _cutout(img)
        if layer is None:
            log.info("rembg unavailable - using 'card' style instead.")
            layer = _card(img)
    elif style == "circle":
        layer = _circle(img)
    else:
        layer = _card(img)

    # Scale to the requested on-screen height (even dimensions keep x264 happy).
    ratio = target_height / layer.height
    layer = layer.resize(
        (
            max(2, int(layer.width * ratio) // 2 * 2),
            max(2, int(layer.height * ratio) // 2 * 2),
        ),
        Image.LANCZOS,
    )
    layer = _add_shadow(
        layer,
        blur=max(10, int(target_height * 0.045)),
        offset=max(5, int(target_height * 0.025)),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    layer.save(out_path, "PNG")
    return out_path, layer.width, layer.height


# --------------------------------------------------------------------------
def _cutout(img: Image.Image) -> Image.Image | None:
    if not settings.enable_cutout:
        return None
    try:
        from rembg import remove  # type: ignore
    except ImportError:
        return None
    try:
        cut = remove(img)
        return cut.convert("RGBA").crop(cut.getbbox() or (0, 0, img.width, img.height))
    except Exception as exc:  # noqa: BLE001 - never let this kill a render
        log.warning("rembg failed (%s) - falling back.", exc)
        return None


def _card(img: Image.Image) -> Image.Image:
    """Portrait crop inside a rounded white frame."""
    target_ratio = 3 / 4  # w:h
    img = _center_crop_to_ratio(img, target_ratio)
    img = img.resize((900, 1200), Image.LANCZOS)

    radius = 48
    border = 18

    canvas = Image.new(
        "RGBA", (img.width + border * 2, img.height + border * 2), (0, 0, 0, 0)
    )
    frame_mask = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(frame_mask).rounded_rectangle(
        [0, 0, canvas.width - 1, canvas.height - 1], radius=radius + border, fill=255
    )
    white = Image.new("RGBA", canvas.size, (255, 255, 255, 255))
    canvas = Image.composite(white, canvas, frame_mask)

    photo_mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(photo_mask).rounded_rectangle(
        [0, 0, img.width - 1, img.height - 1], radius=radius, fill=255
    )
    canvas.paste(img, (border, border), photo_mask)
    return canvas


def _circle(img: Image.Image) -> Image.Image:
    img = _center_crop_to_ratio(img, 1.0).resize((1000, 1000), Image.LANCZOS)

    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, img.width - 1, img.height - 1], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.5))  # antialias the edge

    border = 16
    canvas = Image.new(
        "RGBA", (img.width + border * 2, img.height + border * 2), (0, 0, 0, 0)
    )
    ring = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(ring).ellipse([0, 0, canvas.width - 1, canvas.height - 1], fill=255)
    ring = ring.filter(ImageFilter.GaussianBlur(1.5))
    canvas.paste(Image.new("RGBA", canvas.size, (255, 255, 255, 255)), (0, 0), ring)
    canvas.paste(img, (border, border), mask)
    return canvas


def _center_crop_to_ratio(img: Image.Image, ratio: float) -> Image.Image:
    """Crop to a w:h ratio, biased slightly upward so heads aren't cut off."""
    cur = img.width / img.height
    if cur > ratio:
        new_w = int(img.height * ratio)
        left = (img.width - new_w) // 2
        return img.crop((left, 0, left + new_w, img.height))
    new_h = int(img.width / ratio)
    top = int((img.height - new_h) * 0.35)
    return img.crop((0, top, img.width, top + new_h))


def _add_shadow(layer: Image.Image, blur: int = 28, offset: int = 16) -> Image.Image:
    pad = (blur * 2 + offset) // 2 * 2  # even, so the canvas stays even too
    canvas = Image.new(
        "RGBA", (layer.width + pad * 2, layer.height + pad * 2), (0, 0, 0, 0)
    )

    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    alpha = layer.split()[-1]
    black = Image.new("RGBA", layer.size, (0, 0, 0, 150))
    shadow.paste(black, (pad, pad + offset), alpha)
    shadow = shadow.filter(ImageFilter.GaussianBlur(blur))

    canvas = Image.alpha_composite(canvas, shadow)
    canvas.paste(layer, (pad, pad), layer)
    return canvas
