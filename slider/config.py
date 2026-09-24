"""Configuration loader. Reads config.json (optional); environment variables override.

Every tunable lives here. Other modules must not hardcode cities, keys, folder
IDs, display dimensions, or timing values.
"""

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
        with open(_config_path, "r", encoding="utf-8") as f:
            _cfg = json.load(f)
        if not isinstance(_cfg, dict):
            print("Warning: config.json must contain a JSON object; ignoring it.")
            _cfg = {}
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Warning: Failed to load config.json: {exc}")


def _get(*keys, default=None):
    """Drill into the nested config dict by key path."""
    node = _cfg
    for key in keys:
        if isinstance(node, dict):
            node = node.get(key, None)
        else:
            return default
    return node if node is not None else default


def _num(value, default, cast=float, minimum=None):
    """Coerce a config value to a number, falling back to default on garbage."""
    try:
        result = cast(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and result < minimum:
        return default
    return result


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Display settings
# ---------------------------------------------------------------------------

FRAME_WIDTH: int = _num(_get("display", "width"), 800, int, 64)
FRAME_HEIGHT: int = _num(_get("display", "height"), 480, int, 64)
# SLIDER_TRANSITION_TIME / SLIDER_DISPLAY_TIME env vars override these for quick local runs.
TRANSITION_TIME: float = _num(os.getenv("SLIDER_TRANSITION_TIME"), _num(_get("display", "transition_time"), 2.0, float, 0.0), float, 0.0)
DISPLAY_TIME: float = _num(os.getenv("SLIDER_DISPLAY_TIME"), _num(_get("display", "display_time"), 30.0, float, 1.0), float, 1.0)
FPS: int = _num(_get("display", "fps"), 30, int, 1)
NUM_TRANSITION_FRAMES: int = max(2, int(TRANSITION_TIME * FPS))

# Run in a normal window instead of fullscreen (handy for development on a desktop).
WINDOWED: bool = _truthy(os.getenv("SLIDER_WINDOWED", "")) or _truthy(_get("display", "windowed", default=False))

# Development aids (environment only, never in config.json):
#   SLIDER_DEMO=1            use canned weather/news data instead of the network
#   SLIDER_CAPTURE_DIR=path  save a PNG of the screen every SLIDER_CAPTURE_SECONDS
DEMO_MODE: bool = _truthy(os.getenv("SLIDER_DEMO", ""))
CAPTURE_DIR: str = os.getenv("SLIDER_CAPTURE_DIR", "")
CAPTURE_SECONDS: float = _num(os.getenv("SLIDER_CAPTURE_SECONDS"), 2.0, float, 0.1)

# Local timezone used for the clock, the day/evening schedule, and forecasts.
TIMEZONE: str = os.getenv("SLIDER_TIMEZONE") or _get("timezone", default="America/Chicago")

# Weighted mix of layouts chosen for image slides in "random" mode.
_DEFAULT_MIX = {
    "day_start_hour": 7,
    "day_end_hour": 18,
    "weekday": {"single": 2, "stitch": 2, "quote": 2, "forecast": 2, "today": 2, "news": 6},
    "weekend": {"single": 3, "stitch": 2, "quote": 2, "forecast": 2, "today": 2, "news": 3},
    "evening": {"single": 3, "quote": 2, "today": 2},
}
DISPLAY_MIX: dict = {**_DEFAULT_MIX, **(_get("display_mix", default={}) or {})}

# ---------------------------------------------------------------------------
# Weather settings
# ---------------------------------------------------------------------------

WEATHER_CURRENT_CITY: str = _get("weather", "current_city", default="Waupun")
WEATHER_FORECAST_CITY: str = _get("weather", "forecast_city", default="Fond du Lac")
WEATHER_COUNTRY_CODE: str = _get("weather", "country_code", default="US")

# ---------------------------------------------------------------------------
# Google Drive settings
# ---------------------------------------------------------------------------

DRIVE_FOLDER_ID: str = os.getenv("SLIDER_DRIVE_FOLDER_ID") or _get(
    "google_drive", "folder_id", default="1hpBzZ_kiXpIBtRv1FN3da8zOhT5J0Ggi"
)
DRIVE_SCOPES: list = ["https://www.googleapis.com/auth/drive.readonly"]

# ---------------------------------------------------------------------------
# Camera / brightness settings
# ---------------------------------------------------------------------------

CAMERA_ENABLED: bool = _truthy(_get("camera", "enabled", default=True))
CAMERA_INDEX: int = _num(_get("camera", "index"), 1, int, 0)
BRIGHTNESS_DARK_THRESHOLD: float = _num(_get("camera", "brightness_dark_threshold"), 35.0, float, 0.0)
BRIGHTNESS_CHECK_INTERVAL = timedelta(
    seconds=_num(_get("camera", "brightness_check_interval_seconds"), 15, float, 1.0)
)
# After the camera fails to open, wait this long before trying again.
CAMERA_RETRY_INTERVAL = timedelta(
    seconds=_num(_get("camera", "retry_interval_seconds"), 300, float, 5.0)
)

# ---------------------------------------------------------------------------
# API keys (always from environment)
# ---------------------------------------------------------------------------

WEATHERMAP_API_KEY: str = os.getenv("WEATHERMAP_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY") or ""
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or ""
OPENAI_API_BASE: str = os.getenv("OPENAI_API_BASE") or ""

# ---------------------------------------------------------------------------
# OpenAI model settings (env vars override config.json)
# ---------------------------------------------------------------------------

OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL") or _get("openai", "chat_model", default="gpt-5-mini")
OPENAI_NEWS_MODEL: str = os.getenv("OPENAI_NEWS_MODEL") or _get("openai", "news_model", default="gpt-5-mini")
OPENAI_REQUEST_TIMEOUT: float = _num(_get("openai", "request_timeout"), 60.0, float, 1.0)
# Reasoning effort for models that support it (gpt-5 family, o-series). Empty disables.
OPENAI_REASONING_EFFORT: str = os.getenv("OPENAI_REASONING_EFFORT") or _get("openai", "reasoning_effort", default="low")

# ---------------------------------------------------------------------------
# News settings
# ---------------------------------------------------------------------------

NEWS_TOPICS: list = _get("news", "topics", default=[
    "US politics, economics, and Wall Street",
    "international relations and trade",
    "semiconductors, AI, science, and technology policy",
])
# Optional region for one local story per digest (e.g. "Wisconsin"). Empty disables.
NEWS_LOCAL_AREA: str = _get("news", "local_area", default="")
NEWS_MAX_AGE_HOURS: int = _num(_get("news", "max_age_hours"), 48, int, 1)

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT: float = _num(_get("network", "request_timeout"), 10.0, float, 1.0)

# ---------------------------------------------------------------------------
# Cache TTLs
# ---------------------------------------------------------------------------

MEDIA_REFRESH_INTERVAL = timedelta(minutes=_num(_get("cache_ttl", "media_refresh_minutes"), 2, float, 0.1))
WEATHER_CACHE_TTL = timedelta(minutes=_num(_get("cache_ttl", "weather_minutes"), 15, float, 1))
FORECAST_CACHE_TTL = timedelta(minutes=_num(_get("cache_ttl", "forecast_minutes"), 15, float, 1))
AI_CACHE_SUCCESS_TTL = timedelta(minutes=_num(_get("cache_ttl", "ai_success_minutes"), 30, float, 1))
AI_CACHE_FAILURE_TTL = timedelta(minutes=_num(_get("cache_ttl", "ai_failure_minutes"), 5, float, 0.1))
NEWS_CACHE_FAILURE_TTL = timedelta(minutes=_num(_get("cache_ttl", "news_failure_minutes"), 3, float, 0.1))
NEWS_POOL_REFRESH_TTL = timedelta(hours=_num(_get("cache_ttl", "news_pool_refresh_hours"), 2, float, 0.05))
NEWS_POOL_SIZE: int = _num(_get("cache_ttl", "news_pool_size"), 6, int, 1)
# How often the background thread re-checks the data caches (the TTLs above
# decide when a network call actually happens).
DATA_POLL_INTERVAL = timedelta(seconds=_num(_get("cache_ttl", "data_poll_seconds"), 60, float, 5))

# ---------------------------------------------------------------------------
# Local files
# ---------------------------------------------------------------------------

IMAGES_DIR: str = resource_path("images")
SCALED_DIR: str = os.path.join(IMAGES_DIR, ".scaled")
METADATA_FILE: str = resource_path("metadata.json")
WEATHER_CACHE_FILE: str = resource_path("weather_cache.json")
FORECAST_CACHE_FILE: str = resource_path("forecast_cache.json")
NEWS_CACHE_FILE: str = resource_path("news_cache.json")
TOKEN_FILE: str = resource_path("token.json")
CREDENTIALS_FILE: str = resource_path("credentials.json")
QUOTES_FILE: str = resource_path("quotes.json")
ICONS_DIR: str = resource_path("icons")

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
    """Return True when running on resource-constrained hardware (ARM Linux)."""
    forced = os.getenv("SLIDER_FORCE_LOW_POWER", "").lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False

    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine:
        return machine.startswith(("arm", "aarch64")) or machine in {"arm64", "armv7l", "armv6l"}
    return False


if _low_power_setting == "auto":
    LOW_POWER_MODE: bool = _detect_low_power_device()
else:
    LOW_POWER_MODE = _truthy(_low_power_setting)

# ---------------------------------------------------------------------------
# Media extensions
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

BUTTON_AUTOHIDE_SECONDS: float = _num(_get("ui", "button_autohide_seconds"), 5, float, 1)
MAX_IMAGE_CACHE: int = _num(_get("ui", "max_image_cache"), 12, int, 2)
# Longest edge (pixels) of the pre-scaled display copies of images.
IMAGE_MAX_DIM: int = max(FRAME_WIDTH, FRAME_HEIGHT)
