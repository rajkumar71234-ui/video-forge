FROM python:3.11-slim

# ffmpeg does the rendering; the DejaVu fonts are what libass uses to draw
# burned-in captions. Without the fonts, captions render as empty boxes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        fonts-dejavu-core \
        fonts-noto-core \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flat layout: every source file sits in one directory.
COPY *.py .
COPY *.htm* .

RUN mkdir -p /app/data /app/assets
ENV DATA_DIR=/app/data ASSETS_DIR=/app/assets

EXPOSE 8000

# Render/Railway inject $PORT. Fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
