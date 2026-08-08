"""Tests for Scene mode against a fake Gemini image endpoint.

    python3 test_sc.py

Verifies the request shape, the prompt wording, image framing, error handling,
and that a full Scene job renders a real MP4 end to end without ever calling
Google. What it cannot prove is that the live API likes our payload - only a
run with a real key shows that.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8155
BASE = f"http://127.0.0.1:{PORT}/v1beta"
os.environ.update(
    GEMINI_API_KEY="fake-key",
    GEMINI_API_BASE=BASE,
    IMAGE_PROVIDER="stub",
    ENABLE_CAPTIONS="false",
    VIDEO_WIDTH="854",
    VIDEO_HEIGHT="480",
    VIDEO_PRESET="ultrafast",
)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image  # noqa: E402

failures: list[str] = []
captured: dict = {}
MODE = {"reply": "ok"}


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    # Portrait on purpose: proves the cover-crop reframes to landscape.
    Image.new("RGB", (w, h), (90, 140, 200)).save(buf, "PNG")
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        captured["body"] = json.loads(self.rfile.read(n))
        captured["headers"] = dict(self.headers)

        if MODE["reply"] == "quota":
            body, code = {"error": {"message": "quota"}}, 429
        elif MODE["reply"] == "blocked":
            body, code = {"candidates": [
                {"content": {"parts": [{"text": "I can't help with that."}]}}
            ]}, 200
        else:
            body, code = {"candidates": [{"content": {"parts": [
                {"inlineData": {"mime_type": "image/png",
                                "data": base64.b64encode(_png(600, 900)).decode()}}
            ]}}]}, 200

        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


async def main() -> int:
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    import scene

    tmp = Path("/tmp/vf_sc"); tmp.mkdir(parents=True, exist_ok=True)
    photo = tmp / "me.jpg"
    Image.new("RGB", (900, 1200), (190, 160, 140)).save(photo)

    print("\n== prompt wording ==")
    p = scene.build_prompt("swimming in a blue sea touching a dolphin")
    check("refers to the photo's person", "person from the provided photo" in p)
    check("asks to keep likeness", "recognisable as the same person" in p)
    check("asks for full body in scene", "full-body" in p)
    check("blocks frame/collage artefacts", "photo frame" in p and "collage" in p)
    check("blocks text artefacts", "watermarks" in p)
    check("carries the user's action", "touching a dolphin" in p)

    print("\n== request shape ==")
    out = tmp / "scene.png"
    await scene.generate_scene(photo, "swimming with a dolphin", out, 854, 480)
    b = captured["body"]
    parts = b["contents"][0]["parts"]
    check("sends photo then text", "inline_data" in parts[0] and "text" in parts[1])
    check("photo sent as base64", len(parts[0]["inline_data"]["data"]) > 100)
    check("mime type is image/jpeg",
          parts[0]["inline_data"]["mime_type"] == "image/jpeg")
    check("asks for IMAGE modality",
          b["generationConfig"]["responseModalities"] == ["IMAGE"])
    check("auth header set", captured["headers"].get("x-goog-api-key") == "fake-key")

    print("\n== framing ==")
    im = Image.open(out)
    check("output is exactly the video frame size", im.size == (854, 480), str(im.size))
    check("saved as PNG", im.format == "PNG")

    print("\n== error handling ==")
    MODE["reply"] = "quota"
    try:
        await scene.generate_scene(photo, "x", tmp / "a.png", 854, 480)
        check("quota raises", False)
    except scene.SceneError as e:
        check("quota raises", True)
        check("quota message suggests Composite", "Composite" in str(e), str(e)[:60])

    MODE["reply"] = "blocked"
    try:
        await scene.generate_scene(photo, "x", tmp / "b.png", 854, 480)
        check("refusal raises", False)
    except scene.SceneError as e:
        check("refusal raises", True)
        check("explains public-figure block", "public" in str(e).lower(), str(e)[:60])
    MODE["reply"] = "ok"

    print("\n== missing key is refused clearly ==")
    import config
    saved = config.settings.gemini_api_key
    config.settings.gemini_api_key = ""
    try:
        await scene.generate_scene(photo, "x", tmp / "c.png", 854, 480)
        check("no-key raises", False)
    except scene.SceneError as e:
        check("no-key raises", True)
        check("names the variable", "GEMINI_API_KEY" in str(e))
    config.settings.gemini_api_key = saved

    print("\n== full Scene job end to end ==")
    voice = Path("/tmp/vf/voice.ogg")
    if not voice.exists():
        print("  (skipped: /tmp/vf/voice.ogg fixture missing)")
    else:
        import jobs as jobs_mod
        wd = tmp / "job"; wd.mkdir(exist_ok=True)
        j = jobs_mod.Job(id="sc1", prompt="swimming with a dolphin", mode="scene",
                         workdir=wd, photo_path=photo, voice_path=voice, mood="calm")
        await jobs_mod._process(j)
        check("job completed", j.status == "done", j.error or "")
        if j.status == "done":
            import subprocess
            info = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_streams", "-show_format",
                 "-of", "json", str(j.video_path)],
                capture_output=True, text=True).stdout)
            v = next(s for s in info["streams"] if s["codec_type"] == "video")
            a = [s for s in info["streams"] if s["codec_type"] == "audio"]
            check("has video+audio", bool(a) and v["codec_name"] == "h264")
            check("854x480", (v["width"], v["height"]) == (854, 480),
                  f'{v["width"]}x{v["height"]}')
            check("no photo card was overlaid",
                  not (wd / "subject.png").exists(),
                  "subject.png absent as expected")

    srv.shutdown()
    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    print("\n" + "=" * 52)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("ALL SCENE-MODE CHECKS PASSED")
    sys.exit(code)
