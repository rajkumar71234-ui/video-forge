"""Tests for AI video mode against a fake Gemini Omni server.

Runs a stand-in for generativelanguage.googleapis.com so the request shape and
response parsing can be verified without spending money or needing a key.

    python3 test_ai.py

What this does NOT prove: that the real Google API accepts our payload. Only a
run against the live endpoint with a billed key can show that. It does prove
the request matches the documented schema, and that every documented response
shape (inline base64, URI delivery, safety refusal, quota error) is handled.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

os.environ.setdefault("GEMINI_API_KEY", "fake-key-for-testing")
os.environ.setdefault("ENABLE_VIDGEN", "true")

PORT = 8144
BASE = f"http://127.0.0.1:{PORT}/v1beta"
os.environ["GEMINI_API_BASE"] = BASE

sys.path.insert(0, str(Path(__file__).resolve().parent))

failures: list[str] = []
captured: dict = {}
MODE = {"reply": "inline"}
FAKE_MP4 = b"\x00\x00\x00\x20ftypisom" + b"\xAB" * 40000


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _json(self, code: int, body: dict):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        captured["body"] = json.loads(self.rfile.read(n))
        captured["headers"] = dict(self.headers)

        if MODE["reply"] == "quota":
            return self._json(429, {"error": {"message": "quota"}})
        if MODE["reply"] == "refused":
            return self._json(200, {"status": "completed", "steps": [
                {"type": "model_output", "content": [{"type": "text", "text": "blocked"}]}
            ]})
        if MODE["reply"] == "uri":
            return self._json(200, {"status": "completed", "steps": [
                {"type": "thought", "content": [{"type": "thought", "text": "..."}]},
                {"type": "model_output", "content": [{
                    "type": "video", "mime_type": "video/mp4",
                    "uri": f"{BASE}/files/abc123:download?alt=media",
                }]},
            ]})
        return self._json(200, {"status": "completed", "steps": [
            {"type": "model_output", "content": [{
                "type": "video", "mime_type": "video/mp4",
                "data": base64.b64encode(FAKE_MP4).decode(),
            }]},
        ]})

    def do_GET(self):
        if ":download" in self.path:
            self.send_response(200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(len(FAKE_MP4)))
            self.end_headers()
            self.wfile.write(FAKE_MP4)
            return
        self._json(200, {"state": "ACTIVE", "name": "files/abc123"})


async def main() -> int:
    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    import vidgen

    tmp = Path("/tmp/vf_ai"); tmp.mkdir(parents=True, exist_ok=True)
    photo = tmp / "photo.jpg"
    if not photo.exists():
        from PIL import Image
        Image.new("RGB", (800, 1000), (180, 150, 130)).save(photo)

    print("\n== prompt building ==")
    p = vidgen.build_prompt("is swimming in the sea with a dolphin",
                            "I love you", 8)
    check("includes the action", "swimming in the sea" in p)
    check("quotes the dialogue", '"I love you"' in p)
    check("uses timecodes", "[0-" in p and "s]" in p)
    check("binds the subject reference", "<IMAGE_REF_0>" in p)
    check("suppresses on-screen text", "no captions" in p.lower())

    p2 = vidgen.build_prompt("is walking on a beach", "", 8)
    check("no-dialogue prompt says so", "No dialogue" in p2)
    check("no-dialogue prompt has no quotes", '"' not in p2.replace('""', ''))

    print("\n== request shape (vs Google's documented schema) ==")
    prov = vidgen.OmniVideoProvider()
    out = tmp / "inline.mp4"
    await prov.generate(photo=photo, prompt=p, out_path=out)
    b = captured["body"]
    check("model field sent", b.get("model") == "gemini-omni-flash-preview", str(b.get("model")))
    check("input is a list", isinstance(b.get("input"), list))
    types = [i.get("type") for i in b.get("input", [])]
    check("sends image then text", types == ["image", "text"], str(types))
    img = b["input"][0]
    check("image is base64 data", isinstance(img.get("data"), str) and len(img["data"]) > 100)
    check("image mime_type set", img.get("mime_type") == "image/jpeg", str(img.get("mime_type")))
    vc = b.get("generation_config", {}).get("video_config", {})
    check("task = reference_to_video", vc.get("task") == "reference_to_video", str(vc))
    rf = b.get("response_format", {})
    check("response_format type video", rf.get("type") == "video")
    check("delivery = uri (needed >4MB)", rf.get("delivery") == "uri")
    check("aspect_ratio sent", rf.get("aspect_ratio") == "16:9")
    check("auth via x-goog-api-key header",
          captured["headers"].get("x-goog-api-key") == "fake-key-for-testing")

    print("\n== response handling ==")
    check("inline base64 saved", out.exists() and out.stat().st_size == len(FAKE_MP4),
          f"{out.stat().st_size} bytes")

    MODE["reply"] = "uri"
    out2 = tmp / "uri.mp4"
    await prov.generate(photo=photo, prompt=p, out_path=out2)
    check("uri delivery downloads", out2.exists() and out2.stat().st_size == len(FAKE_MP4),
          f"{out2.stat().st_size} bytes")

    MODE["reply"] = "refused"
    try:
        await prov.generate(photo=photo, prompt=p, out_path=tmp / "x.mp4")
        check("safety refusal raises", False)
    except vidgen.VideoGenError as e:
        check("safety refusal raises", True)
        check("refusal message mentions public figures",
              "public figure" in str(e).lower(), str(e)[:70])

    MODE["reply"] = "quota"
    try:
        await prov.generate(photo=photo, prompt=p, out_path=tmp / "x.mp4")
        check("quota error raises", False)
    except vidgen.VideoGenError as e:
        check("quota error raises", True)
        check("quota message mentions billing", "billing" in str(e).lower(), str(e)[:70])

    print("\n== guard rails ==")
    import config
    saved = config.settings.enable_vidgen
    config.settings.enable_vidgen = False
    import jobs as jobs_mod
    j = jobs_mod.Job(id="t", prompt="x", mode="aigen", workdir=tmp, photo_path=photo)
    await jobs_mod._process(j)
    check("aigen refused when ENABLE_VIDGEN=false", j.status == "error")
    check("error explains how to enable",
          "ENABLE_VIDGEN" in (j.error or ""), (j.error or "")[:70])
    config.settings.enable_vidgen = saved

    srv.shutdown()
    return 0


if __name__ == "__main__":
    code = asyncio.run(main())
    print("\n" + "=" * 52)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("ALL AI-MODE CHECKS PASSED")
    sys.exit(code)
