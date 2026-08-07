"""End-to-end smoke test.

Boots the real app, posts a real photo + voice note, waits for the render, and
verifies the resulting MP4 actually has a video stream, an audio stream and
roughly the expected duration.

Run it before every deploy:

    IMAGE_PROVIDER=stub python3 test_e2e.py

Uses the stub image provider by default so it costs nothing and works offline.
Set IMAGE_PROVIDER=gemini with a real key to also exercise the API path.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

os.environ.setdefault("IMAGE_PROVIDER", "stub")
os.environ.setdefault("ENABLE_CAPTIONS", "false")
os.environ.setdefault("VIDEO_WIDTH", "1280")
os.environ.setdefault("VIDEO_HEIGHT", "720")

HERE = Path(__file__).parent
FIXTURES = Path(os.getenv("FIXTURE_DIR", "/tmp/vf"))
BASE = "http://127.0.0.1:8099"

failures: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    return json.loads(out)


def main() -> int:
    photo = FIXTURES / "photo.jpg"
    voice = FIXTURES / "voice.ogg"
    if not photo.exists() or not voice.exists():
        print(f"Missing fixtures in {FIXTURES}. Expected photo.jpg and voice.ogg.")
        return 2

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app",
         "--host", "127.0.0.1", "--port", "8099", "--log-level", "warning"],
        cwd=str(HERE), env=os.environ.copy(),
    )
    try:
        print("\n== health ==")
        for _ in range(40):
            try:
                r = httpx.get(f"{BASE}/api/health", timeout=2)
                if r.status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        else:
            print("server never came up")
            return 2

        health = r.json()
        check("health endpoint ok", health.get("ok") is True, health.get("provider", ""))

        print("\n== ui served ==")
        idx = httpx.get(f"{BASE}/", timeout=10)
        check("index.html served", idx.status_code == 200 and "Video" in idx.text)

        print("\n== validation ==")
        bad = httpx.post(
            f"{BASE}/api/jobs",
            data={"prompt": "hello there"},
            files={
                "photo": ("notes.txt", b"nope", "text/plain"),
                "voice": ("voice.ogg", voice.read_bytes(), "audio/ogg"),
            },
            timeout=30,
        )
        check("rejects a bad photo type", bad.status_code == 400, str(bad.status_code))

        print("\n== submit ==")
        resp = httpx.post(
            f"{BASE}/api/jobs",
            data={
                "prompt": "A warm sunrise over a calm coastline, cinematic wide shot",
                "style": "card",
                "mood": "calm",
                "motion": "zoom_in",
                "bgm_volume": "0.22",
                "captions": "false",
            },
            files={
                "photo": ("photo.jpg", photo.read_bytes(), "image/jpeg"),
                "voice": ("voice.ogg", voice.read_bytes(), "audio/ogg"),
            },
            timeout=60,
        )
        check("job accepted", resp.status_code == 202, str(resp.status_code))
        if resp.status_code != 202:
            print(resp.text[:500])
            return 1
        job_id = resp.json()["id"]

        print("\n== render ==")
        deadline = time.time() + 300
        state = {}
        while time.time() < deadline:
            state = httpx.get(f"{BASE}/api/jobs/{job_id}", timeout=10).json()
            print(f"    {state['progress']:3d}%  {state['step']}")
            if state["status"] in ("done", "error"):
                break
            time.sleep(2)

        check("render completed", state.get("status") == "done", state.get("error") or "")
        if state.get("status") != "done":
            return 1

        print("\n== output ==")
        vid = httpx.get(f"{BASE}{state['video_url']}", timeout=120)
        check("video downloads", vid.status_code == 200, f"{len(vid.content)} bytes")
        out = Path("/tmp/vf/out.mp4")
        out.write_bytes(vid.content)

        info = probe(out)
        streams = info["streams"]
        v = next((s for s in streams if s["codec_type"] == "video"), None)
        a = next((s for s in streams if s["codec_type"] == "audio"), None)
        dur = float(info["format"]["duration"])

        check("has a video stream", v is not None, v["codec_name"] if v else "")
        check("has an audio stream", a is not None, a["codec_name"] if a else "")
        check("h264 video", v and v["codec_name"] == "h264")
        check("aac audio", a and a["codec_name"] == "aac")
        check("resolution 1280x720", v and (v["width"], v["height"]) == (1280, 720),
              f"{v['width']}x{v['height']}" if v else "")
        check("yuv420p (plays everywhere)", v and v.get("pix_fmt") == "yuv420p")
        # voice is 8s + 1.2s tail
        check("duration ~9.2s", 8.8 <= dur <= 9.8, f"{dur:.2f}s")
        check("file is not trivially small", out.stat().st_size > 50_000,
              f"{out.stat().st_size} bytes")

        # Audio must not be silent - proves the mix actually landed.
        vol = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", str(out), "-af", "volumedetect",
             "-f", "null", "-"],
            capture_output=True, text=True,
        ).stderr
        mean = next(
            (float(l.split("mean_volume:")[1].split("dB")[0])
             for l in vol.splitlines() if "mean_volume:" in l),
            None,
        )
        check("audio is not silent", mean is not None and mean > -50,
              f"mean {mean} dB" if mean is not None else "no reading")

        print("\n== thumbnail ==")
        th = httpx.get(f"{BASE}/api/jobs/{job_id}/thumb", timeout=30)
        check("thumbnail served", th.status_code == 200, f"{len(th.content)} bytes")

        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    code = main()
    print("\n" + "=" * 52)
    if failures:
        print(f"FAILED ({len(failures)}): " + ", ".join(failures))
        sys.exit(1)
    print("ALL CHECKS PASSED" if code == 0 else f"ABORTED (exit {code})")
    sys.exit(code)
