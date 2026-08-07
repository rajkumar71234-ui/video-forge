# Video Forge

Upload a **photo**, a **voice note**, and a **prompt** → get a finished MP4 with an
AI-generated background, cinematic camera motion, your narration, and background
music that automatically ducks under the voice.

FastAPI + FFmpeg + Google Gemini, in one Docker container.

---

## Why every file is in one folder

There are no subfolders here, on purpose. GitHub's drag-and-drop uploader
flattens nested directories, which silently breaks Python packages. Keeping the
layout flat makes the project immune to that: you can drag every file in at once
and it just works.

| File | Role |
|---|---|
| `main.py` | FastAPI routes, uploads, serves the UI |
| `jobs.py` | job queue + render orchestration |
| `config.py` | all settings, driven by environment variables |
| `providers.py` | image backends (Gemini, offline stub) |
| `compose.py` | the FFmpeg filter graph — the actual render |
| `subject.py` | photo → framed / circle / cut-out layer |
| `bgm.py` | synthesized royalty-free music |
| `captions.py` | Gemini transcription → SRT |
| `ffmpeg_utils.py` | ffmpeg/ffprobe wrappers |
| `index.html` | the entire UI, one file |
| `Dockerfile` / `render.yaml` | deployment |
| `test_e2e.py` | end-to-end test |

---

## What it does

```
photo.jpg  ─┐
voice.ogg  ─┼─►  Gemini: prompt → background image
"prompt"   ─┘         │
                      ├─►  Pillow: photo → composited layer
                      │
                      └─►  FFmpeg, single pass:
                             • Ken Burns move on the background
                             • photo centred with a slow float + shadow
                             • optional burned-in captions
                             • voice normalised to −16 LUFS
                             • music sidechain-ducked under the voice
                             • fade in/out → H.264 + AAC MP4
```

Video length comes from the voice note, plus a 1.2 s tail.

---

## Deploying to Render

1. Put these files in a GitHub repo (drag them all into the repo's upload page).
2. Render → **New → Blueprint** → pick the repo. It reads `render.yaml`.
3. It deploys in **free test mode** — backgrounds are generated locally as
   gradients, so it works immediately with no API key and no cost.
4. When you want real AI backgrounds: get a key at
   <https://aistudio.google.com/apikey>, then in Render → Environment set
   `GEMINI_API_KEY` to your key and `IMAGE_PROVIDER` to `gemini`.

## Running locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Open <http://127.0.0.1:8000>. Requires `ffmpeg` on PATH.

## Testing

```bash
IMAGE_PROVIDER=stub python3 test_e2e.py
```

Boots the real server, submits a real job, and asserts the MP4 has an H.264
stream, AAC audio, the right resolution, `yuv420p`, the expected duration, and
non-silent audio.

---

## Free-tier realities

- Render's free plan sleeps after 15 min idle; first request then takes ~50 s.
- Free = 0.1 CPU, so `render.yaml` ships at **720p**. Raise `VIDEO_WIDTH` /
  `VIDEO_HEIGHT` to 1920/1080 on a paid plan.
- Disk is ephemeral. Jobs are temporary by design and swept after
  `JOB_TTL_MINUTES`.
- Cost with a real API key is roughly **₹2–4 per video**. Music is free — it's
  synthesized by FFmpeg at render time, so there's no licensing question.

## Known limitations

- **The photo doesn't lip-sync.** This is a composite with motion, not an
  avatar. Talking-head animation needs a service like D-ID or HeyGen, which
  would slot in as another provider in `providers.py`.
- **Caption timing is approximate.** Gemini's audio timestamps drift on longer
  recordings. Swap `captions.py` for `faster-whisper` if you need accuracy.
- **No auth.** Anyone with the URL can use your Gemini quota. Add a key check in
  `main.py` before sharing it publicly.
- **Jobs live in memory.** A restart loses in-flight jobs.
