"""Central configuration. Everything is driven by environment variables.

Fields whose env-var name differs from the attribute name declare an explicit
`validation_alias`. Without it, pydantic-settings would look for an env var
matching the attribute name (e.g. WIDTH instead of VIDEO_WIDTH) and silently
ignore anything set in a .env file.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---- AI provider -------------------------------------------------
    # "gemini" = real AI backgrounds (needs a key)
    # "stub"   = procedural gradients rendered locally: no key, no cost.
    image_provider: str = Field("gemini", validation_alias="IMAGE_PROVIDER")
    gemini_api_key: str = Field("", validation_alias="GEMINI_API_KEY")
    gemini_image_model: str = Field(
        "gemini-2.5-flash-image", validation_alias="GEMINI_IMAGE_MODEL"
    )
    gemini_text_model: str = Field(
        "gemini-2.5-flash", validation_alias="GEMINI_TEXT_MODEL"
    )
    gemini_api_base: str = Field(
        "https://generativelanguage.googleapis.com/v1beta",
        validation_alias="GEMINI_API_BASE",
    )

    # ---- AI video generation (beta) ----------------------------------
    # Off by default: every generated clip costs real money, so it must be
    # switched on deliberately rather than discovered by accident.
    enable_vidgen: bool = Field(False, validation_alias="ENABLE_VIDGEN")
    omni_model: str = Field(
        "gemini-omni-flash-preview", validation_alias="OMNI_MODEL"
    )
    # Generation blocks for a long time; this bounds it so a stuck job fails
    # loudly instead of sitting at 70% forever.
    vidgen_timeout_sec: float = Field(600.0, validation_alias="VIDGEN_TIMEOUT_SEC")
    vidgen_seconds: int = Field(8, validation_alias="VIDGEN_SECONDS")

    # ---- Storage -----------------------------------------------------
    data_dir: Path = Field(BASE_DIR / "data", validation_alias="DATA_DIR")
    assets_dir: Path = Field(BASE_DIR / "assets", validation_alias="ASSETS_DIR")
    # Finished jobs are deleted after this long, so disk doesn't creep up.
    job_ttl_minutes: int = Field(120, validation_alias="JOB_TTL_MINUTES")

    # ---- Limits ------------------------------------------------------
    max_upload_mb: int = Field(25, validation_alias="MAX_UPLOAD_MB")
    max_duration_sec: int = Field(180, validation_alias="MAX_DURATION_SEC")
    # 1 is right for a small box: ffmpeg already saturates the CPU, so
    # parallel renders make everything slower rather than faster.
    max_concurrent_renders: int = Field(1, validation_alias="MAX_CONCURRENT_RENDERS")

    # ---- Video -------------------------------------------------------
    width: int = Field(1920, validation_alias="VIDEO_WIDTH")
    height: int = Field(1080, validation_alias="VIDEO_HEIGHT")
    fps: int = Field(30, validation_alias="VIDEO_FPS")
    preset: str = Field("veryfast", validation_alias="VIDEO_PRESET")
    crf: int = Field(23, validation_alias="VIDEO_CRF")

    # ---- Features ----------------------------------------------------
    # Requires rembg + onnxruntime (see requirements.txt).
    enable_cutout: bool = Field(False, validation_alias="ENABLE_CUTOUT")
    enable_captions: bool = Field(True, validation_alias="ENABLE_CAPTIONS")


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.assets_dir.mkdir(parents=True, exist_ok=True)
