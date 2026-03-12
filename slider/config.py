"""Configuration loader. Reads config.json (optional), env vars override."""

import json
import os
import platform
from datetime import timedelta

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resource_path(*parts: str) -> str:
    """Resolve a path relative to the project root directory."""
    return os.path.join(SCRIPT_DIR, *parts)


# ---------------------------------------------------------------------------
# Load config.json (optional — all values have defaults)
# ---------------------------------------------------------------------------

_config_path = resource_path("config.json")
_cfg = {}
if os.path.exists(_config_path):
    try:
        with open(_config_path, "r") as f:
            _cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Failed to load config.json: {exc}")


def _get(*keys, default=None):
    """Drill into nested config dict by key path."""
    node = _cfg
    for key in keys:
        if isinstance(node, dict):
            node = node.get(key, None)
        else:
            return default
    return node if node is not None else default


# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------

FRAME_WIDTH: int = _get("display", "width", default=800)
FRAME_HEIGHT: int = _get("display", "height", default=480)
TRANSITION_TIME: int = _get("display", "transition_time", default=2)
DISPLAY_TIME: int = _get("display", "display_time", default=30)
FPS: int = _get("display", "fps", default=30)
NUM_TRANSITION_FRAMES: int = int(TRANSITION_TIME * FPS)

# ---------------------------------------------------------------------------
# Weather settings
# ---------------------------------------------------------------------------

WEATHER_CURRENT_CITY: str = _get("weather", "current_city", default="Waupun")
WEATHER_FORECAST_CITY: str = _get("weather", "forecast_city", default="Fond du Lac")
WEATHER_COUNTRY_CODE: str = _get("weather", "country_code", default="US")

# ---------------------------------------------------------------------------
# Google Drive settings
# ---------------------------------------------------------------------------

DRIVE_FOLDER_ID: str = _get("google_drive", "folder_id", default="1hpBzZ_kiXpIBtRv1FN3da8zOhT5J0Ggi")
DRIVE_SCOPES: list = ["https://www.googleapis.com/auth/drive.readonly"]

# ---------------------------------------------------------------------------
# Camera / brightness settings
# ---------------------------------------------------------------------------

CAMERA_INDEX: int = _get("camera", "index", default=1)
BRIGHTNESS_DARK_THRESHOLD: float = _get("camera", "brightness_dark_threshold", default=35.0)
BRIGHTNESS_CHECK_INTERVAL = timedelta(
    seconds=_get("camera", "brightness_check_interval_seconds", default=15)
)

# ---------------------------------------------------------------------------
# API keys (always from environment)
# ---------------------------------------------------------------------------

WEATHERMAP_API_KEY: str = os.getenv("WEATHERMAP_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY") or ""
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or ""
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# ---------------------------------------------------------------------------
# OpenAI model settings (env vars override config.json)
# ---------------------------------------------------------------------------

OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL") or _get("openai", "chat_model", default="gpt-4o-mini")
OPENAI_NEWS_MODEL: str = os.getenv("OPENAI_NEWS_MODEL") or _get("openai", "news_model", default="gpt-4o-mini")
OPENAI_REQUEST_TIMEOUT: int = _get("openai", "request_timeout", default=30)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT: int = 10  # seconds, for weather/Drive API calls

# ---------------------------------------------------------------------------
# Cache TTLs
# ---------------------------------------------------------------------------

MEDIA_REFRESH_INTERVAL = timedelta(minutes=_get("cache_ttl", "media_refresh_minutes", default=2))
AI_CACHE_SUCCESS_TTL = timedelta(minutes=_get("cache_ttl", "ai_success_minutes", default=30))
AI_CACHE_FAILURE_TTL = timedelta(minutes=_get("cache_ttl", "ai_failure_minutes", default=5))
FORECAST_CACHE_TTL = timedelta(minutes=_get("cache_ttl", "forecast_minutes", default=15))
NEWS_CACHE_SUCCESS_TTL = timedelta(minutes=_get("cache_ttl", "news_success_minutes", default=15))
NEWS_CACHE_FAILURE_TTL = timedelta(minutes=_get("cache_ttl", "news_failure_minutes", default=3))
NEWS_POOL_REFRESH_TTL = timedelta(hours=_get("cache_ttl", "news_pool_refresh_hours", default=2))
NEWS_POOL_SIZE: int = _get("cache_ttl", "news_pool_size", default=6)
NEWS_CACHE_FILE: str = resource_path("news_cache.json")

# ---------------------------------------------------------------------------
# Touch mode definitions
# ---------------------------------------------------------------------------

MODE_DEFINITIONS: list = _get("modes", default=[
    {"mode": "random", "label": "Random"},
    {"mode": "news", "label": "News"},
    {"mode": "weather", "label": "Weather"},
    {"mode": "pictures", "label": "Pics"},
    {"mode": "video", "label": "Video"},
])

# ---------------------------------------------------------------------------
# Low power mode
# ---------------------------------------------------------------------------

_low_power_setting = _get("low_power_mode", default="auto")


def _detect_low_power_device() -> bool:
    """Return True when running on resource-constrained hardware (ARM)."""
    forced = os.getenv("SLIDER_FORCE_LOW_POWER", "").lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine:
        arm_like = (
            machine.startswith("arm")
            or machine.startswith("aarch64")
            or machine in {"arm64", "armv7l", "armv6l"}
        )
        if arm_like:
            return True

    return False


if _low_power_setting == "auto":
    LOW_POWER_MODE: bool = _detect_low_power_device()
elif _low_power_setting in (True, "true", "yes", "on", "1"):
    LOW_POWER_MODE = True
else:
    LOW_POWER_MODE = False

# ---------------------------------------------------------------------------
# Media extensions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

BUTTON_AUTOHIDE_SECONDS: int = 5
MAX_IMAGE_CACHE: int = 10
