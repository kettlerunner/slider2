from __future__ import print_function

import os
import io
import json
import random
import time
import traceback
import math
import platform
import ctypes
from ctypes import c_char_p, c_ulong, c_void_p
from ctypes.util import find_library
from datetime import datetime, timedelta
from collections import defaultdict
import re
import textwrap

import cv2
import numpy as np
import requests
from requests import RequestException

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None


# ---------------------------------------------------------------------------
# Basic config / paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts: str) -> str:
    """Resolve a path relative to the script directory."""
    return os.path.join(SCRIPT_DIR, *parts)


def sanitize_text(text: str) -> str:
    """Replace curly quotes/dashes and strip non-ASCII characters."""
    replacements = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
        "…": "...",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    return text


def get_central_time() -> datetime:
    """Return current time in America/Chicago, falling back to local."""
    if ZoneInfo is not None:
        try:
            return datetime.now(ZoneInfo("America/Chicago"))
        except Exception:
            pass
    return datetime.now()


# Frame / timing settings
frame_width = 800
frame_height = 480
transition_time = 2
display_time = 30
num_transition_frames = int(transition_time * 30)

# API keys
api_key = os.getenv("WEATHERMAP_API_KEY") or os.getenv("OPENWEATHERMAP_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2-mini")
OPENAI_NEWS_MODEL = os.getenv("OPENAI_NEWS_MODEL", OPENAI_CHAT_MODEL)

# Touch mode controls
MODE_DEFINITIONS = [
    {"mode": "random", "label": "Random"},
    {"mode": "news", "label": "News"},
    {"mode": "weather", "label": "Weather"},
    {"mode": "pictures", "label": "Pics"},
    {"mode": "video", "label": "Video"},
]

# Screen dimming based on ambient brightness (0-255 grayscale)
BRIGHTNESS_DARK_THRESHOLD = 35.0          # tweak this up/down based on your room
BRIGHTNESS_CHECK_INTERVAL = timedelta(seconds=15)  # how often to sample the camera

def _detect_low_power_device() -> bool:
    """Return True when running on resource constrained hardware."""
    forced = os.getenv("SLIDER_FORCE_LOW_POWER", "").lower()
    if forced in {"1", "true", "yes", "on"}:
        return True
    if forced in {"0", "false", "no", "off"}:
        return False

    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux" and machine:
        arm_like = (
            machine.startswith("arm"),
            machine.startswith("aarch64"),
            machine in {"arm64", "armv7l", "armv6l"},
        )
        if any(arm_like):
            return True

    return False


#LOW_POWER_MODE = _detect_low_power_device()
LOW_POWER_MODE = False
_LOW_POWER_NOTICE_SHOWN = False

REQUEST_TIMEOUT = 10  # seconds

MEDIA_REFRESH_INTERVAL = timedelta(minutes=2)
AI_CACHE_SUCCESS_TTL = timedelta(minutes=30)
AI_CACHE_FAILURE_TTL = timedelta(minutes=5)
FORECAST_CACHE_TTL = timedelta(minutes=15)
NEWS_CACHE_SUCCESS_TTL = timedelta(minutes=15)
NEWS_CACHE_FAILURE_TTL = timedelta(minutes=3)
NEWS_CACHE_FILE = resource_path("news_cache.json")
NEWS_POOL_SIZE = 6
NEWS_POOL_REFRESH_TTL = timedelta(hours=2)


# ---------------------------------------------------------------------------
# OpenAI client + helpers
# ---------------------------------------------------------------------------

if OpenAI is not None and openai_key:
    try:
        client = OpenAI(api_key=openai_key)
    except Exception as exc:
        print(f"Failed to initialize OpenAI client: {exc}")
        client = None
else:
    client = None


def _safe_chat_completion(model: str, messages, timeout: int = REQUEST_TIMEOUT):
    """
    Call OpenAI chat completions with a hard timeout.

    If the SDK doesn't accept the timeout argument, fall back to a background
    thread and join with the same timeout.
    """
    if client is None:
        return None

    # Preferred path: modern SDK with timeout support
    try:
        return client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
        )
    except TypeError:
        # Older SDK: try without timeout, but guard with a thread
        pass
    except Exception as exc:
        print(f"OpenAI request failed: {exc}")
        return None

    # Fallback: manual timeout using a background thread
    import threading

    result_container = {}

    def _request():
        try:
            result_container["value"] = client.chat.completions.create(
                model=model,
                messages=messages,
            )
        except Exception as exc:
            result_container["error"] = exc

    thread = threading.Thread(target=_request, daemon=True)
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print(f"OpenAI request exceeded {timeout} seconds and was aborted.")
        return None

    if "error" in result_container:
        print(f"OpenAI request failed: {result_container['error']}")
        return None

    return result_container.get("value")


# ---------------------------------------------------------------------------
# Google Drive / auth
# ---------------------------------------------------------------------------

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Supported media extensions
image_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
video_extensions = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def authenticate_drive():
    """Authenticate to Google Drive using credentials/token stored next to script."""
    creds = None
    token_path = resource_path("token.json")
    creds_path = resource_path("credentials.json")

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, SCOPES)
        except Exception as exc:
            print(f"Failed to load existing credentials: {exc}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                print(f"Failed to refresh credentials: {exc}")
                creds = None
        else:
            if not os.path.exists(creds_path):
                print(f"Missing credentials.json at {creds_path}.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as exc:
                print(f"Failed to authenticate with Google Drive: {exc}")
                return None
        try:
            with open(token_path, "w") as token:
                token.write(creds.to_json())
        except OSError as exc:
            print(f"Failed to save credentials: {exc}")

    try:
        return build("drive", "v3", credentials=creds)
    except Exception as exc:
        print(f"Failed to build Google Drive service: {exc}")
        return None


def list_files_in_folder(service, folder_id):
    if service is None:
        return None
    query = f"'{folder_id}' in parents"
    try:
        results = (
            service.files()
            .list(q=query, pageSize=200, fields="nextPageToken, files(id, name, modifiedTime, size)")
            .execute()
        )
    except HttpError as exc:
        print(f"Failed to list files: {exc}")
        return None
    except Exception as exc:
        print(f"Unexpected error listing files: {exc}")
        return None

    return results.get("files", [])


def download_file(service, file_id, file_name):
    if service is None:
        return False
    try:
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(file_name, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
    except HttpError as exc:
        print(f"Failed to download file {file_id}: {exc}")
        return False
    except OSError as exc:
        print(f"Failed to write file {file_name}: {exc}")
        return False
    except Exception as exc:
        print(f"Unexpected error downloading file {file_id}: {exc}")
        return False
    return True


def parse_modified_time(modified_time_str):
    if not modified_time_str:
        return datetime.min
    try:
        if modified_time_str.endswith("Z"):
            modified_time_str = modified_time_str[:-1] + "+00:00"
        return datetime.fromisoformat(modified_time_str)
    except ValueError:
        return datetime.min


def load_local_metadata(metadata_file):
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to load metadata: {exc}")
    return {}


def save_local_metadata(metadata_file, metadata):
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    except OSError as exc:
        print(f"Failed to save metadata: {exc}")


def load_media_from_local_cache(temp_dir):
    """Load media items that were previously downloaded to disk (paths only)."""
    if not os.path.isdir(temp_dir):
        return []

    media_items = []
    for entry in sorted(os.listdir(temp_dir)):
        file_path = os.path.join(temp_dir, entry)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(entry)[1].lower()
        if ext in image_extensions:
            media_type = "image"
        elif ext in video_extensions:
            media_type = "video"
        else:
            continue

        media_items.append(
            {
                "type": media_type,
                "path": file_path,
                "name": entry,
                "modifiedTime": None,
            }
        )

    return media_items


def refresh_media_items(service, folder_id, temp_dir, metadata_file, local_metadata):
    """
    Sync media from Google Drive.

    Returns (media_items, updated_metadata, downloaded_files) or
    (None, local_metadata, set()) on error.
    """
    if service is None:
        return None, local_metadata, set()

    files = list_files_in_folder(service, folder_id)
    if files is None:
        print("Skipping media refresh due to retrieval error.")
        return None, local_metadata, set()

    if not files:
        # Folder has no files
        save_local_metadata(metadata_file, {})
        return [], {}, set()

    os.makedirs(temp_dir, exist_ok=True)

    sorted_files = sorted(
        files,
        key=lambda item: parse_modified_time(item.get("modifiedTime")),
        reverse=True,
    )

    updated_metadata = {}
    media_items = []
    downloaded_files = set()

    for file in sorted_files:
        file_name = file["name"]
        file_path = os.path.join(temp_dir, file_name)
        remote_size = int(file.get("size", 0) or 0)
        file_metadata = {
            "modifiedTime": file.get("modifiedTime"),
            "size": remote_size,
        }

        ext = os.path.splitext(file_name)[1].lower()
        if ext not in image_extensions + video_extensions:
            continue

        needs_download = False
        local_file_metadata = local_metadata.get(file_name)
        if local_file_metadata is None:
            needs_download = True
        else:
            local_size = int(local_file_metadata.get("size", 0) or 0)
            if (
                local_file_metadata.get("modifiedTime") != file_metadata["modifiedTime"]
                or local_size != file_metadata["size"]
            ):
                needs_download = True
            elif not os.path.exists(file_path):
                needs_download = True

        if needs_download:
            print(f"Downloading file: {file_name}")
            if not download_file(service, file["id"], file_path):
                print(f"Skipping file due to download error: {file_name}")
                continue
            downloaded_files.add(file_name)

        if not os.path.exists(file_path):
            print(f"File missing after download: {file_name}")
            continue

        media_type = "video" if ext in video_extensions else "image"
        media_items.append(
            {
                "type": media_type,
                "path": file_path,
                "name": file_name,
                "modifiedTime": file_metadata["modifiedTime"],
            }
        )

        updated_metadata[file_name] = file_metadata

    save_local_metadata(metadata_file, updated_metadata)

    return media_items, updated_metadata, downloaded_files


# ---------------------------------------------------------------------------
# Cursor / window utilities
# ---------------------------------------------------------------------------

_window_fullscreen_state = {}

_cursor_state = {
    "hidden": False,
    "system": platform.system(),
    "display": None,
    "window": None,
    "x11": None,
    "xfixes": None,
}


def hide_mouse_cursor():
    """Attempt to hide the system mouse cursor while the slideshow is active."""
    state = _cursor_state
    if state["hidden"]:
        return

    system = state["system"]

    try:
        if system == "Windows":
            ctypes.windll.user32.ShowCursor(False)
            state["hidden"] = True
        elif system == "Darwin":
            try:
                from AppKit import NSCursor  # type: ignore
            except ImportError:
                return
            NSCursor.hide()
            state["hidden"] = True
        elif system == "Linux":
            lib_x11 = find_library("X11")
            lib_xfixes = find_library("Xfixes")
            if not lib_x11 or not lib_xfixes:
                return

            x11 = ctypes.cdll.LoadLibrary(lib_x11)
            xfixes = ctypes.cdll.LoadLibrary(lib_xfixes)

            x11.XOpenDisplay.argtypes = [c_char_p]
            x11.XOpenDisplay.restype = c_void_p
            display = x11.XOpenDisplay(None)
            if not display:
                return

            x11.XDefaultRootWindow.argtypes = [c_void_p]
            x11.XDefaultRootWindow.restype = c_ulong
            root = x11.XDefaultRootWindow(display)

            xfixes.XFixesHideCursor.argtypes = [c_void_p, c_ulong]
            xfixes.XFixesHideCursor.restype = None
            xfixes.XFixesHideCursor(display, root)

            x11.XFlush.argtypes = [c_void_p]
            x11.XFlush.restype = None
            x11.XFlush(display)

            state.update(
                {
                    "hidden": True,
                    "display": display,
                    "window": root,
                    "x11": x11,
                    "xfixes": xfixes,
                }
            )
    except Exception as exc:
        print(f"Failed to hide mouse cursor: {exc}")


def show_mouse_cursor():
    """Restore the system mouse cursor after the slideshow ends."""
    state = _cursor_state
    if not state["hidden"]:
        return

    system = state["system"]

    try:
        if system == "Windows":
            ctypes.windll.user32.ShowCursor(True)
        elif system == "Darwin":
            try:
                from AppKit import NSCursor  # type: ignore
            except ImportError:
                pass
            else:
                NSCursor.unhide()
        elif system == "Linux":
            display = state.get("display")
            window = state.get("window")
            x11 = state.get("x11")
            xfixes = state.get("xfixes")
            if display and window is not None and x11 and xfixes:
                xfixes.XFixesShowCursor.argtypes = [c_void_p, c_ulong]
                xfixes.XFixesShowCursor.restype = None
                xfixes.XFixesShowCursor(display, window)

                x11.XFlush.argtypes = [c_void_p]
                x11.XFlush.restype = None
                x11.XFlush(display)

                x11.XCloseDisplay.argtypes = [c_void_p]
                x11.XCloseDisplay.restype = ctypes.c_int
                x11.XCloseDisplay(display)
    except Exception as exc:
        print(f"Failed to restore mouse cursor: {exc}")
    finally:
        state.update(
            {
                "hidden": False,
                "display": None,
                "window": None,
                "x11": None,
                "xfixes": None,
            }
        )


def ensure_fullscreen(window_name: str):
    """Ensure the OpenCV window stays in fullscreen mode."""
    try:
        fullscreen_flag = getattr(cv2, "WINDOW_FULLSCREEN", 1)
        window_state = _window_fullscreen_state.setdefault(window_name, {"is_fullscreen": False})

        try:
            property_value = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            is_fullscreen = int(property_value) == fullscreen_flag
        except cv2.error:
            is_fullscreen = False

        if not is_fullscreen:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, fullscreen_flag)
            cv2.moveWindow(window_name, 0, 0)
            cv2.resizeWindow(window_name, frame_width, frame_height)
            window_state["is_fullscreen"] = True
        elif not window_state.get("is_fullscreen", False):
            window_state["is_fullscreen"] = True
    except cv2.error as exc:
        print(f"Failed to enforce fullscreen for '{window_name}': {exc}")


def normalize_frame_for_display(frame, enforce_size=True):
    """Normalize frame data so every render is consistent for fullscreen playback."""
    if frame is None:
        return None

    normalized = frame

    if not isinstance(normalized, np.ndarray):
        return None

    if normalized.dtype != np.uint8:
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    if normalized.ndim == 2:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    elif normalized.ndim == 3:
        channels = normalized.shape[2]
        if channels == 1:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGRA2BGR)

    if enforce_size and (
        normalized.shape[0] != frame_height or normalized.shape[1] != frame_width
    ):
        normalized = cv2.resize(normalized, (frame_width, frame_height), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(normalized)


def show_frame(window_name: str, frame):
    """Display a frame and re-assert fullscreen."""
    prepared_frame = normalize_frame_for_display(frame)
    if prepared_frame is None:
        return
    cv2.imshow(window_name, prepared_frame)
    ensure_fullscreen(window_name)

BUTTON_AUTOHIDE_SECONDS = 5


def _update_button_visibility(ui_state):
    if not ui_state:
        return
    if not ui_state.get("show_buttons"):
        return
    last_touch = ui_state.get("last_touch")
    if last_touch is None:
        return
    if time.monotonic() - last_touch >= BUTTON_AUTOHIDE_SECONDS:
        ui_state["show_buttons"] = False


def present_frame(frame, temp, weather, status_text=None, ambient_dark=False, ui_state=None):
    """
    Final step before putting pixels on the screen.

    If ambient_dark is True, we show a black screen instead of the frame
    (simple "screen off" behavior when the lights are out).
    Otherwise, we overlay time/weather and show the frame as usual.
    """
    _update_button_visibility(ui_state)
    if ambient_dark:
        black = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        show_frame("slideshow", black)
    else:
        overlay = add_time_overlay(frame, temp, weather, status_text=status_text)
        if ui_state and ui_state.get("show_buttons") and overlay is not None:
            overlay = draw_mode_buttons(overlay, ui_state["buttons"], ui_state["mode"])
        show_frame("slideshow", overlay)


# ---------------------------------------------------------------------------
# Weather + news
# ---------------------------------------------------------------------------

def get_weather_forecast2(
    api_key, city="Fond du Lac", country_code="US", cache_file=None
):
    """Fetches today's detailed forecast (3h slices) and caches it."""
    if cache_file is None:
        cache_file = resource_path("forecast_cache.json")

    if not api_key:
        print("OpenWeatherMap API key is missing.")
        return []

    cached_forecast = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as cache_handle:
                cache_payload = json.load(cache_handle)
            timestamp = datetime.strptime(cache_payload["timestamp"], "%Y-%m-%d %H:%M:%S")
            cached_forecast = cache_payload.get("forecast", [])
            if datetime.now() - timestamp < FORECAST_CACHE_TTL:
                return cached_forecast
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"Failed to read forecast cache: {exc}")
            cached_forecast = None

    url = (
        f"http://api.openweathermap.org/data/2.5/forecast"
        f"?q={city},{country_code}&units=imperial&appid={api_key}"
    )
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except RequestException as exc:
        print(f"Error fetching weather data: {exc}")
        return cached_forecast or []

    if response.status_code == 200:
        data = response.json()
        today = datetime.now().strftime("%Y-%m-%d")
        today_weather = []

        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"])
            if dt.strftime("%Y-%m-%d") == today:
                today_weather.append(
                    {
                        "time": dt.strftime("%I:%M %p"),
                        "temp": item["main"]["temp"],
                        "description": item["weather"][0]["description"],
                        "wind_speed": item["wind"]["speed"],
                        "humidity": item["main"]["humidity"],
                    }
                )

        try:
            with open(cache_file, "w") as cache_handle:
                json.dump(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "forecast": today_weather,
                    },
                    cache_handle,
                )
        except OSError as exc:
            print(f"Failed to write forecast cache: {exc}")

        return today_weather

    print("Error fetching weather data.")
    return cached_forecast or []


def get_weather_forecast(api_key, city="Fond du Lac", country_code="US"):
    """Return a 5-day (min/max) forecast."""
    if not api_key:
        print("OpenWeatherMap API key is missing.")
        return None
    url = (
        f"http://api.openweathermap.org/data/2.5/forecast"
        f"?q={city},{country_code}&units=imperial&appid={api_key}"
    )
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except RequestException as exc:
        print(f"Error fetching weather forecast data: {exc}")
        return None

    if response.status_code == 200:
        data = response.json()
        daily_forecast = defaultdict(
            lambda: {"temp_min": float("inf"), "temp_max": float("-inf"), "descriptions": []}
        )

        for item in data.get("list", []):
            dt = datetime.fromtimestamp(item["dt"])
            date_key = dt.strftime("%Y-%m-%d")

            daily_forecast[date_key]["temp_min"] = min(
                daily_forecast[date_key]["temp_min"], item["main"]["temp_min"]
            )
            daily_forecast[date_key]["temp_max"] = max(
                daily_forecast[date_key]["temp_max"], item["main"]["temp_max"]
            )
            daily_forecast[date_key]["descriptions"].append(item["weather"][0]["description"])

            if dt.hour == 12:
                daily_forecast[date_key]["main_description"] = item["weather"][0]["description"]

        formatted_forecast = []
        for date, forecast in daily_forecast.items():
            dt = datetime.strptime(date, "%Y-%m-%d")
            descriptions = forecast["descriptions"]
            dominant_desc = (
                forecast.get("main_description")
                if "main_description" in forecast
                else max(set(descriptions), key=descriptions.count)
            )
            formatted_forecast.append(
                {
                    "date": dt.strftime("%A, %b %d"),
                    "temp_min": forecast["temp_min"],
                    "temp_max": forecast["temp_max"],
                    "description": dominant_desc,
                }
            )

        formatted_forecast.sort(key=lambda x: datetime.strptime(x["date"], "%A, %b %d"))
        return formatted_forecast[:5]
    else:
        print("Error fetching weather forecast data")
        return None


icon_images = {
    "clear": cv2.imread(resource_path("icons", "sunny.png"), cv2.IMREAD_UNCHANGED),
    "cloudy": cv2.imread(resource_path("icons", "cloudy.png"), cv2.IMREAD_UNCHANGED),
    "rain": cv2.imread(resource_path("icons", "rain.png"), cv2.IMREAD_UNCHANGED),
    "snow": cv2.imread(resource_path("icons", "snow.png"), cv2.IMREAD_UNCHANGED),
    "windy": cv2.imread(resource_path("icons", "windy.png"), cv2.IMREAD_UNCHANGED),
}


def get_weather_icon(description):
    """Return the appropriate icon image based on the weather description."""
    description = description.lower()
    if "clear" in description:
        return icon_images.get("clear")
    elif "cloud" in description:
        return icon_images.get("cloudy")
    elif "rain" in description:
        return icon_images.get("rain")
    elif "wind" in description or "breeze" in description:
        return icon_images.get("windy")
    elif "snow" in description:
        return icon_images.get("snow")
    return None


_news_cache = {
    "expires": datetime.min,
    "pool": [
        {
            "headline": "News unavailable",
            "summary": "No updates retrieved yet.",
            "sources": [],
            "bias": "Center",
            "bias_note": "",
        }
    ],
    "status": "failure",
}


def _load_news_cache_from_disk():
    if not os.path.exists(NEWS_CACHE_FILE):
        return None
    try:
        with open(NEWS_CACHE_FILE, "r") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read news cache: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    timestamp_str = payload.get("timestamp")
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None

    pool = payload.get("pool")
    value = payload.get("value")
    status = payload.get("status", "failure")
    if isinstance(pool, list) and pool:
        normalized_pool = []
        for entry in pool:
            normalized = _normalize_news_payload(entry)
            if normalized:
                normalized_pool.append(normalized)
        if not normalized_pool:
            return None
        pool = normalized_pool
    elif isinstance(value, dict):
        normalized = _normalize_news_payload(value)
        if not normalized:
            return None
        pool = [normalized]
    else:
        return None

    return {
        "timestamp": timestamp,
        "pool": pool,
        "status": status if status in {"success", "failure"} else "failure",
    }


def _save_news_cache_to_disk(pool, status):
    try:
        first_value = pool[0] if isinstance(pool, list) and pool else None
        with open(NEWS_CACHE_FILE, "w") as handle:
            json.dump(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "pool": pool,
                    "value": first_value,
                    "status": status,
                },
                handle,
            )
    except OSError as exc:
        print(f"Failed to write news cache: {exc}")


def _safe_responses_create(model: str, prompt: str, timeout: int = REQUEST_TIMEOUT):
    """Call OpenAI Responses API with optional web search support."""
    if client is None:
        return None

    if not hasattr(client, "responses"):
        print("OpenAI client does not support Responses API. Update the openai package.")
        return None

    import threading

    def _attempt_call(args):
        try:
            return client.responses.create(**args, timeout=timeout), None
        except TypeError:
            pass
        except Exception as exc:
            return None, exc

        result_container = {}

        def _request():
            try:
                result_container["value"] = client.responses.create(**args)
            except Exception as exc:
                result_container["error"] = exc

        thread = threading.Thread(target=_request, daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            print(f"OpenAI responses request exceeded {timeout} seconds and was aborted.")
            return None, TimeoutError("OpenAI responses request timed out.")

        if "error" in result_container:
            return None, result_container["error"]

        return result_container.get("value"), None

    attempts = [
        {"model": model, "input": prompt, "tools": [{"type": "web_search"}]},
        {"model": model, "input": prompt, "tools": [{"type": "web_search_preview"}]},
        {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search"}],
        },
        {
            "model": model,
            "input": [{"role": "user", "content": prompt}],
            "tools": [{"type": "web_search_preview"}],
        },
    ]

    last_error = None
    for args in attempts:
        response, error = _attempt_call(args)
        if response is not None:
            return response
        if error is not None:
            last_error = error

    if last_error is not None:
        print(f"OpenAI responses request failed: {last_error}")
    return None


def _extract_response_text(response):
    if response is None:
        return ""
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output_items = getattr(response, "output", None)
    if isinstance(output_items, list):
        chunks = []
        for item in output_items:
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text = getattr(content, "text", "")
                    if text:
                        chunks.append(text)
        if chunks:
            return "\n".join(chunks).strip()
    return ""


def _normalize_news_payload(parsed):
    if not isinstance(parsed, dict):
        return None

    headline = sanitize_text(str(parsed.get("headline", "")).strip())
    summary = sanitize_text(str(parsed.get("summary", "")).strip())

    sources = []
    raw_sources = parsed.get("sources", [])
    if isinstance(raw_sources, list):
        for entry in raw_sources:
            name = None
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("source") or entry.get("publisher")
            elif isinstance(entry, str):
                name = entry
            if name:
                cleaned = sanitize_text(str(name).strip())
                if cleaned:
                    sources.append(cleaned)

    bias_label = parsed.get("bias_label") or parsed.get("bias") or "center"
    bias_label = str(bias_label).strip().lower()
    bias_map = {"left": "Left", "center": "Center", "right": "Right"}
    bias = bias_map.get(bias_label, "Center")
    bias_note = sanitize_text(str(parsed.get("bias_note", "")).strip())

    if not headline or not summary:
        return None

    return {
        "headline": headline,
        "summary": summary,
        "sources": sources,
        "bias": bias,
        "bias_note": bias_note,
    }


def _normalize_news_pool(parsed):
    if isinstance(parsed, dict) and isinstance(parsed.get("stories"), list):
        stories = parsed.get("stories")
    elif isinstance(parsed, list):
        stories = parsed
    elif isinstance(parsed, dict):
        normalized = _normalize_news_payload(parsed)
        return [normalized] if normalized else None
    else:
        return None

    normalized_pool = []
    for entry in stories:
        normalized = _normalize_news_payload(entry)
        if normalized:
            normalized_pool.append(normalized)
        if len(normalized_pool) >= NEWS_POOL_SIZE:
            break

    if not normalized_pool:
        return None

    return normalized_pool


def get_ai_generated_news():
    """Retrieve a short AI-generated news blurb with aggressive timeouts."""
    prompt = """
Use web search to find six current news stories from the last 48 hours.
Focus on: US politics/economics/Wall Street, international relations/trade, or
semiconductor/AI/science/technology policy. Avoid pop culture, celebrity, or
gossip topics. Gather at least 3 similar articles from different outlets so
the summary avoids political bias. Summarize neutrally in 1-2 sentences.

Return JSON ONLY:
{
  "stories": [
    {
      "headline": "A concise headline",
      "summary": "Neutral 1-2 sentence summary",
      "sources": [
        {"name": "Outlet name", "url": "https://example.com/article"}
      ],
      "bias_label": "left|center|right",
      "bias_note": "Short note on why this is the label (e.g., mixed sources)."
    }
  ]
}
    """.strip()

    fallback_value = {
        "headline": "News unavailable",
        "summary": "Unable to retrieve the latest update.",
        "sources": [],
        "bias": "Center",
        "bias_note": "",
    }

    now = datetime.now()
    cache_entry = _news_cache
    if now < cache_entry["expires"] and cache_entry.get("pool"):
        return random.choice(cache_entry["pool"])

    disk_cache = _load_news_cache_from_disk()
    if disk_cache:
        age = now - disk_cache["timestamp"]
        ttl = NEWS_POOL_REFRESH_TTL if disk_cache["status"] == "success" else NEWS_CACHE_FAILURE_TTL
        if age < ttl:
            _news_cache.update(
                {
                    "expires": now + ttl,
                    "pool": disk_cache["pool"],
                    "status": disk_cache["status"],
                }
            )
            return random.choice(disk_cache["pool"])

    if client is None:
        _news_cache.update(
            {
                "expires": now + NEWS_CACHE_FAILURE_TTL,
                "pool": [fallback_value],
                "status": "failure",
            }
        )
        return fallback_value

    response = _safe_responses_create(model=OPENAI_NEWS_MODEL, prompt=prompt)
    news_data = _extract_response_text(response)

    if news_data:
        if news_data.startswith("```json"):
            news_data = news_data[len("```json") :].strip()
        if news_data.endswith("```"):
            news_data = news_data[: -len("```")].strip()

        try:
            parsed = json.loads(news_data)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse AI news response: {exc}")
        else:
            normalized_pool = _normalize_news_pool(parsed)
            if normalized_pool:
                _news_cache.update(
                    {
                        "expires": now + NEWS_POOL_REFRESH_TTL,
                        "pool": normalized_pool,
                        "status": "success",
                    }
                )
                _save_news_cache_to_disk(normalized_pool, "success")
                return random.choice(normalized_pool)
            print("AI news response missing required fields. Using fallback.")
    else:
        print("AI news request failed or returned no content. Using fallback.")

    fallback_pool = [fallback_value]
    if disk_cache and isinstance(disk_cache.get("pool"), list) and disk_cache["pool"]:
        fallback_pool = disk_cache["pool"]

    _news_cache.update(
        {
            "expires": now + NEWS_CACHE_FAILURE_TTL,
            "pool": fallback_pool,
            "status": "failure",
        }
    )
    _save_news_cache_to_disk(fallback_pool, "failure")
    return random.choice(fallback_pool)


def get_weather_data(api_key, cache_file=None):
    """Return (temp, weather) for Waupun, WI."""
    if cache_file is None:
        cache_file = resource_path("weather_cache.json")

    if not api_key:
        print("OpenWeatherMap API key is missing.")
        return None, None

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            timestamp = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")
            if datetime.now() - timestamp < timedelta(minutes=15):
                return data["temp"], data["weather"]
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"Failed to read weather cache: {exc}")

    url = (
        "http://api.openweathermap.org/data/2.5/weather"
        "?q=Waupun,WI,US&units=imperial&appid={}".format(api_key)
    )
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except RequestException as exc:
        print(f"Error fetching weather data: {exc}")
        return None, None

    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        weather = data["weather"][0]["main"]
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "temp": temp,
                        "weather": weather,
                    },
                    f,
                )
        except OSError as exc:
            print(f"Failed to write weather cache: {exc}")
        return temp, weather

    print("Error fetching weather data")
    return None, None


# ---------------------------------------------------------------------------
# AI weather summary cache
# ---------------------------------------------------------------------------

_forecast_summary_cache = {
    "key": None,
    "expires": datetime.min,
    "value": ("Weather summary unavailable.", "random"),
}

style_titles = {
    "poem": "Today's Forecast in Verse",
    "haiku": "Today's Haiku Forecast",
    "zen_master": "Today's Zencast",
}


def get_tldr_forecast(weather_data, style="random"):
    """Generate a concise forecast summary using OpenAI."""
    if not weather_data:
        return "Weather summary unavailable.", style, False

    formatted_data = "\n".join(
        f"Time: {item['time']}, Temp: {item['temp']}°F, Description: {item['description']}, "
        f"Wind Speed: {item['wind_speed']} mph, Humidity: {item['humidity']}%"
        for item in weather_data
    )

    styles = ["poem", "haiku", "zen_master"]
    chosen_style = random.choice(styles) if style == "random" else style

    style_prompts = {
        "poem": "Turn the weather forecast into a very short whimsical poem. Be concise.",
        "haiku": "Write the weather forecast as a tiny haiku. Minimal and poetic.",
        "zen_master": "Summarize the weather like a Zen master. Very concise, direct, profound.",
    }

    prompt = f"""
Here is the weather forecast for today:

{formatted_data}

{style_prompts.get(chosen_style, "Summarize this into a conversational, super short forecast.")}
    """.strip()

    fallback_message = "Weather summary unavailable."

    if client is None:
        return fallback_message, chosen_style, False

    completion = _safe_chat_completion(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )

    if completion and completion.choices:
        try:
            summary = completion.choices[0].message.content.strip()
        except (AttributeError, IndexError):
            summary = ""
        if summary:
            return sanitize_text(summary), chosen_style, True

    print(f"Error generating forecast summary in {chosen_style} style. Using fallback.")
    return fallback_message, chosen_style, False


def get_or_generate_forecast_summary(weather_data, style="random"):
    """Cached wrapper around get_tldr_forecast."""
    if not weather_data:
        return "Weather summary unavailable.", "random"

    cache_key = json.dumps(weather_data, sort_keys=True)
    now = datetime.now()
    cache = _forecast_summary_cache

    if cache["key"] == cache_key and now < cache["expires"]:
        return cache["value"]

    summary, style_used, success = get_tldr_forecast(weather_data, style=style)
    ttl = AI_CACHE_SUCCESS_TTL if success else AI_CACHE_FAILURE_TTL

    cache.update(
        {
            "key": cache_key,
            "value": (summary, style_used),
            "expires": now + ttl,
        }
    )
    return summary, style_used


# ---------------------------------------------------------------------------
# Image helpers / transitions
# ---------------------------------------------------------------------------

def resize_and_pad(image, width, height):
    """Resize an image to fit in (width,height) with black padding."""
    if image is None:
        return None

    img = image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(width / float(w), height / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    channels = resized.shape[2] if resized.ndim == 3 else 3
    if channels == 4:
        padded = np.zeros((height, width, 4), dtype=np.uint8)
    else:
        padded = np.zeros((height, width, 3), dtype=np.uint8)

    top_pad = (height - resized.shape[0]) // 2
    left_pad = (width - resized.shape[1]) // 2

    padded[top_pad : top_pad + resized.shape[0], left_pad : left_pad + resized.shape[1]] = resized
    return padded


def create_zoomed_blurred_background(image, width, height):
    img = image.copy()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / float(w), height / float(h)) * 1.1
    new_w, new_h = int(w * scale), int(h * scale)
    zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    start_x = max((zoomed.shape[1] - width) // 2, 0)
    start_y = max((zoomed.shape[0] - height) // 2, 0)
    cropped = zoomed[start_y : start_y + height, start_x : start_x + width]

    blurred = cv2.GaussianBlur(cropped, (31, 31), 0)
    return blurred


def create_blurred_background(image, width, height):
    img = image.copy()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / float(w), height / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    start_x = max((resized.shape[1] - width) // 2, 0)
    start_y = max((resized.shape[0] - height) // 2, 0)
    centered = resized[start_y : start_y + height, start_x : start_x + width]

    blurred = cv2.GaussianBlur(centered, (31, 31), 0)
    return blurred


def ensure_same_channels(img1, img2):
    """Ensure both images have the same number of channels."""
    if img1 is None or img2 is None:
        return img1, img2

    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if img1.ndim == 3 and img2.ndim == 3 and img1.shape[2] != img2.shape[2]:
        if img1.shape[2] == 3 and img2.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
        elif img1.shape[2] == 4 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

    return img1, img2


def _transition_alphas(num_frames):
    if num_frames <= 1:
        return [1.0]
    return np.linspace(0, 1, num_frames, endpoint=True)


def fade_transition(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    for alpha in _transition_alphas(num_frames):
        blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
        yield blended


def slide_transition_left(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dx = int(width * alpha)
        frame = np.zeros_like(current_img)
        if dx < width:
            frame[:, : width - dx] = current_img[:, dx:]
        if dx > 0:
            frame[:, width - dx :] = next_img[:, :dx]
        yield frame


def slide_transition_right(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dx = int(width * alpha)
        frame = np.zeros_like(current_img)
        if dx < width:
            frame[:, dx:] = current_img[:, : width - dx]
        if dx > 0:
            frame[:, :dx] = next_img[:, width - dx :]
        yield frame


def wipe_transition_top(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dy = int(height * alpha)
        frame = np.zeros_like(current_img)
        if dy < height:
            frame[: height - dy, :] = current_img[dy:, :]
        if dy > 0:
            frame[height - dy :, :] = next_img[:dy, :]
        yield frame


def wipe_transition_bottom(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dy = int(height * alpha)
        frame = np.zeros_like(current_img)
        if dy < height:
            frame[dy:, :] = current_img[: height - dy, :]
        if dy > 0:
            frame[:dy, :] = next_img[height - dy :, :]
        yield frame


# Advanced transitions (only used when not in low-power mode)

def melt_transition(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    max_shift = int(height * 0.5)

    for alpha in _transition_alphas(num_frames):
        frame = next_img.copy().astype(np.float32)

        for r in range(height):
            shift = int(alpha * max_shift)
            new_r = r + shift
            if new_r < height:
                row_alpha = 1 - alpha
                curr_row = current_img[r, :, :].astype(np.float32)

                if curr_row.shape[1] == 4:
                    curr_bgr = curr_row[:, :3]
                    curr_a = curr_row[:, 3] / 255.0
                    curr_a = curr_a[:, np.newaxis]
                    base = frame[new_r, :, :3]
                    out_bgr = curr_bgr * curr_a * row_alpha + base * (1 - curr_a * row_alpha)
                    frame[new_r, :, :3] = out_bgr
                else:
                    curr_bgr = curr_row[:, :3]
                    base = frame[new_r, :, :3]
                    out_bgr = base * (1 - row_alpha) + curr_bgr * row_alpha
                    frame[new_r, :, :3] = out_bgr

        yield frame.astype(np.uint8)


def wave_transition(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    max_vertical_shift = int(height * 0.4)
    max_horizontal_shift = int(width * 0.02)

    for alpha in _transition_alphas(num_frames):
        frame = next_img.copy().astype(np.float32)
        row_alpha = 1 - alpha

        for r in range(height):
            vertical_phase = (r / float(height)) * 2 * np.pi
            vertical_shift = int(np.sin(vertical_phase + alpha * 2 * np.pi) * max_vertical_shift * alpha)

            horizontal_phase = (r / float(height)) * 4 * np.pi
            horizontal_shift = int(np.sin(horizontal_phase + alpha * 4 * np.pi) * max_horizontal_shift * alpha)

            new_r = r + vertical_shift
            if 0 <= new_r < height:
                curr_row = current_img[r, :, :].astype(np.float32)
                has_alpha = curr_row.shape[1] == 4

                if has_alpha:
                    curr_bgr = curr_row[:, :3]
                    curr_a = (curr_row[:, 3] / 255.0) * row_alpha
                    curr_a = curr_a[:, np.newaxis]
                else:
                    curr_bgr = curr_row[:, :3]
                    curr_a = row_alpha

                shifted_bgr = np.roll(curr_bgr, horizontal_shift, axis=0)
                if has_alpha:
                    shifted_a = np.roll(curr_a, horizontal_shift, axis=0)
                else:
                    shifted_a = curr_a

                base = frame[new_r, :, :3]

                if has_alpha:
                    out_bgr = shifted_bgr * shifted_a + base * (1 - shifted_a)
                else:
                    out_bgr = base * (1 - shifted_a) + shifted_bgr * shifted_a

                frame[new_r, :, :3] = out_bgr

        yield frame.astype(np.uint8)


def zen_ripple_transition(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)

    def smoothstep(t):
        return 3 * t ** 2 - 2 * t ** 3

    ys, xs = np.indices((height, width))
    distances = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)

    for alpha in _transition_alphas(num_frames):
        radius = alpha * max_radius
        blend_region = 10
        lower_bound = radius - blend_region
        upper_bound = radius + blend_region

        curr = current_img.astype(np.float32)
        nxt = next_img.astype(np.float32)
        frame = nxt.copy()

        factor = np.zeros_like(distances, dtype=np.float32)
        inside = distances < lower_bound
        outside = distances > upper_bound
        blend_zone = ~inside & ~outside

        factor[inside] = 1.0
        if np.any(blend_zone):
            blend_zone_dist = (distances[blend_zone] - lower_bound) / (upper_bound - lower_bound)
            blend_zone_factor = 1.0 - blend_zone_dist
            blend_zone_factor = smoothstep(blend_zone_factor)
            factor[blend_zone] = blend_zone_factor
        factor[outside] = 0.0

        factor_3c = factor[:, :, np.newaxis]
        blended = curr * (1 - factor_3c) + nxt * factor_3c
        yield blended.astype(np.uint8)


def dynamic_petal_bloom_transition(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0

    N = 8
    max_rotation = math.radians(30)
    scale_factor = 0.3
    inward_factor = 0.2
    blend_boundary = math.radians(2)

    def smoothstep(t):
        return 3 * t ** 2 - 2 * t ** 3

    ys, xs = np.indices((height, width))
    dx = xs - center_x
    dy = ys - center_y
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    angle_norm = (angle + 2 * math.pi) % (2 * math.pi)

    petal_angle = 2 * math.pi / N
    petal_index = (angle_norm // petal_angle).astype(np.int32)

    petal_center_angle = petal_index * petal_angle + petal_angle / 2.0
    angle_diff = angle_norm - petal_center_angle
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    for alpha in _transition_alphas(num_frames):
        frame = next_img.copy().astype(np.float32)

        radius_factor = 1 - inward_factor + alpha * (inward_factor + scale_factor)
        wave = (math.cos(math.pi * (1 - alpha)) + 1) / 2.0
        adjusted_radius_factor = radius_factor * (0.9 + 0.1 * wave)

        rot = alpha * max_rotation
        top_half = angle_diff > 0
        half_sign = np.ones_like(angle_diff, dtype=np.float32)
        half_sign[~top_half] = -1.0
        new_angle = angle + half_sign * rot
        new_radius = radius * adjusted_radius_factor

        half_petal = petal_angle / 2.0
        boundary_dist = np.abs(angle_diff) - (half_petal - blend_boundary)
        boundary_mask = np.ones_like(angle_diff, dtype=np.float32)
        in_blend_zone = boundary_dist > 0
        if np.any(in_blend_zone):
            blend_norm = boundary_dist[in_blend_zone] / blend_boundary
            blend_norm = np.clip(blend_norm, 0, 1)
            blend_val = smoothstep(1 - blend_norm)
            boundary_mask[in_blend_zone] = blend_val

        src_x = (new_radius * np.cos(new_angle) + center_x).astype(np.float32)
        src_y = (new_radius * np.sin(new_angle) + center_y).astype(np.float32)

        inside = (src_x >= 0) & (src_x < width) & (src_y >= 0) & (src_y < height)

        src_xi = np.clip(np.round(src_x[inside]).astype(np.int32), 0, width - 1)
        src_yi = np.clip(np.round(src_y[inside]).astype(np.int32), 0, height - 1)

        old_pixels = current_img[src_yi, src_xi].astype(np.float32)
        factor = (1 - alpha)
        bm = boundary_mask[inside, np.newaxis]
        final_factor = factor * bm

        base = frame[inside, :3]
        blended_rgb = base * (1 - final_factor) + old_pixels[:, :3] * final_factor
        frame[inside, :3] = blended_rgb

        yield frame.astype(np.uint8)


def stitch_images(images, width, height):
    """
    Build a multi-photo collage on top of a zoomed, blurred background.

    - Uses one of the images as the blur source for the full-screen background.
    - Tiles up to len(images) photos in a grid.
    - Each tile is scaled with a margin so you see blur between/around them.
    - Handles 3-channel (BGR) and 4-channel (BGRA) images.
    """
    if not images:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # --- 1) Build blurred background from a random image ---
    bg_src = random.choice(images)
    if bg_src is None or bg_src.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    if bg_src.ndim == 2:
        bg_base = cv2.cvtColor(bg_src, cv2.COLOR_GRAY2BGR)
    elif bg_src.ndim == 3 and bg_src.shape[2] == 4:
        bg_base = cv2.cvtColor(bg_src, cv2.COLOR_BGRA2BGR)
    else:
        bg_base = bg_src.copy()

    background = create_zoomed_blurred_background(bg_base, width, height)

    # --- 2) Compute a grid layout (rows x cols) ---
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(float(n) / cols))

    cell_w = width // cols
    cell_h = height // rows
    margin_factor = 0.9  # leave some blurred space around each tile

    # --- 3) Place each image into its cell with alpha-aware blending ---
    for idx, img in enumerate(images):
        if img is None or img.size == 0:
            continue

        # Normalize to BGR or BGRA
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] > 4:
            img = img[:, :, :4]

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue

        # Scale to fit inside cell with margin
        scale = min((cell_w * margin_factor) / float(w),
                    (cell_h * margin_factor) / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Compute cell position
        row = idx // cols
        col = idx % cols
        cell_x = col * cell_w
        cell_y = row * cell_h

        # Center the resized image inside its cell
        left = cell_x + (cell_w - new_w) // 2
        top = cell_y + (cell_h - new_h) // 2

        if left < 0 or top < 0 or left + new_w > width or top + new_h > height:
            # Safety clip if rounding goes weird
            left = max(0, left)
            top = max(0, top)
            new_w = min(new_w, width - left)
            new_h = min(new_h, height - top)
            resized = resized[:new_h, :new_w]

        roi = background[top:top + new_h, left:left + new_w]

        # Blend with alpha if present
        if resized.ndim == 3 and resized.shape[2] == 4:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR).astype(np.float32)
            alpha = resized[:, :, 3].astype(np.float32) / 255.0
            alpha = alpha[:, :, np.newaxis]

            roi_f = roi.astype(np.float32)
            blended = roi_f * (1.0 - alpha) + rgb * alpha
            background[top:top + new_h, left:left + new_w] = blended.astype(np.uint8)
        else:
            background[top:top + new_h, left:left + new_w] = resized

    return background.astype(np.uint8)


def create_single_image_with_background(image, width, height):
    """
    Create a zoomed, blurred background from the image and place
    a slightly smaller version of the image in the center.

    This ensures that any letterbox/pillarbox area is blurred image,
    not solid black bars.
    """
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    # Use a BGR version to build the background
    if image.ndim == 3 and image.shape[2] == 4:
        bg_base = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim == 2:
        bg_base = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        bg_base = image.copy()

    # Full-screen blurred background
    background = create_zoomed_blurred_background(bg_base, width, height)

    # Prepare the foreground image (keep alpha if present)
    img = image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] > 4:
        img = img[:, :, :4]

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return background

    # Scale image so it fits inside the screen with a margin
    # (e.g. 90% of the screen in each dimension)
    margin_factor = 0.9
    scale = min((width * margin_factor) / float(w),
                (height * margin_factor) / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (height - new_h) // 2
    left = (width - new_w) // 2

    # Blend or paste the resized image onto the blurred background
    if resized.ndim == 3 and resized.shape[2] == 4:
        # BGRA: use alpha blending
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]

        roi = background[top:top + new_h, left:left + new_w].astype(np.float32)
        blended = roi * (1.0 - alpha) + rgb.astype(np.float32) * alpha
        background[top:top + new_h, left:left + new_w] = blended.astype(np.uint8)
    else:
        # No alpha: direct copy
        background[top:top + new_h, left:left + new_w] = resized

    return background.astype(np.uint8)


def add_forecast_overlay(frame, forecast):
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 224), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "5-Day Forecast", (10, 30), font, 0.8, font_color, 2, cv2.LINE_AA)

        if forecast:
            col_width = frame.shape[1] // 5
            for i, day in enumerate(forecast):
                x = i * col_width + 10
                y = 60

                cv2.putText(
                    frame,
                    day["date"].split(",")[0],
                    (x, y),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                    cv2.LINE_AA,
                )
                y += 25
                cv2.putText(
                    frame,
                    day["date"].split(",")[1].strip(),
                    (x, y),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                    cv2.LINE_AA,
                )
                y += 35

                temp_text = f"{day['temp_min']:.1f} - {day['temp_max']:.1f} F"
                cv2.putText(
                    frame,
                    temp_text,
                    (x, y),
                    font,
                    font_scale,
                    font_color,
                    thickness,
                    cv2.LINE_AA,
                )
                y += 35

                icon_img = get_weather_icon(day["description"])
                if icon_img is not None:
                    icon_size = 64
                    icon_img = cv2.resize(icon_img, (icon_size, icon_size))
                    icon_y_offset = y - 20
                    icon_x_offset = x + 20

                    if (
                        0 <= icon_y_offset < frame.shape[0] - icon_size
                        and 0 <= icon_x_offset < frame.shape[1] - icon_size
                    ):
                        if icon_img.ndim == 3 and icon_img.shape[2] == 4:
                            alpha = icon_img[:, :, 3] / 255.0
                            alpha = alpha[:, :, np.newaxis]
                            for c in range(3):
                                frame[
                                    icon_y_offset : icon_y_offset + icon_size,
                                    icon_x_offset : icon_x_offset + icon_size,
                                    c,
                                ] = (
                                    icon_img[:, :, c] * alpha[:, :, 0]
                                    + frame[
                                        icon_y_offset : icon_y_offset + icon_size,
                                        icon_x_offset : icon_x_offset + icon_size,
                                        c,
                                    ]
                                    * (1 - alpha[:, :, 0])
                                )
                        else:
                            frame[
                                icon_y_offset : icon_y_offset + icon_size,
                                icon_x_offset : icon_x_offset + icon_size,
                            ] = icon_img

                    y += 64

                desc = day["description"]
                cv2.putText(
                    frame,
                    desc,
                    (x, y),
                    font,
                    font_scale * 0.9,
                    font_color,
                    thickness,
                    cv2.LINE_AA,
                )
        else:
            cv2.putText(
                frame,
                "Forecast data unavailable",
                (10, 100),
                font,
                font_scale,
                font_color,
                thickness,
                cv2.LINE_AA,
            )

        return frame
    except Exception as e:
        print(f"Error adding forecast overlay: {e}")
        return frame


def add_news_overlay(frame, news):
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        font_color = (35, 35, 35)
        thickness = 1
        accent_color = (45, 90, 160)
        title_color = (20, 55, 110)
        panel_color = (245, 245, 245)

        overlay = frame.copy()
        bar_height = _get_status_bar_height()
        y_start = 0
        panel_bottom = max(frame.shape[0] - bar_height, 0)
        cv2.rectangle(
            overlay,
            (0, y_start),
            (frame.shape[1], panel_bottom),
            panel_color,
            -1,
        )
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(
            frame,
            (0, y_start),
            (6, panel_bottom),
            accent_color,
            -1,
        )

        headline = textwrap.wrap(news.get("headline", ""), width=60)
        summary = textwrap.wrap(news.get("summary", ""), width=60)
        sources = news.get("sources", []) or []
        if isinstance(sources, list):
            sources_line = ", ".join(sources[:4])
        else:
            sources_line = ""
        bias_label = news.get("bias", "Center")
        bias_note = news.get("bias_note", "")
        bias_text = f"Bias: {bias_label}"
        if bias_note:
            bias_text = f"{bias_text} ({bias_note})"
        y = y_start + 28

        cv2.putText(
            frame,
            "News Update:",
            (18, y),
            font,
            font_scale,
            title_color,
            thickness + 1,
            cv2.LINE_AA,
        )
        y += 12
        cv2.line(frame, (18, y), (frame.shape[1] - 18, y), (200, 200, 200), 1)
        y += 22

        for line in headline:
            cv2.putText(
                frame,
                line,
                (18, y),
                font,
                font_scale,
                font_color,
                thickness + 1,
                cv2.LINE_AA,
            )
            y += 30

        for line in summary:
            cv2.putText(
                frame,
                line,
                (18, y),
                font,
                font_scale * 0.85,
                font_color,
                thickness,
                cv2.LINE_AA,
            )
            y += 25

        if sources_line:
            for line in textwrap.wrap(f"Sources: {sources_line}", width=70):
                cv2.putText(
                    frame,
                    line,
                    (18, y),
                    font,
                    font_scale * 0.75,
                    font_color,
                    thickness,
                    cv2.LINE_AA,
                )
                y += 22

        if bias_text:
            for line in textwrap.wrap(bias_text, width=70):
                cv2.putText(
                    frame,
                    line,
                    (18, y),
                    font,
                    font_scale * 0.75,
                    font_color,
                    thickness,
                    cv2.LINE_AA,
                )
                y += 22

        return frame
    except Exception as e:
        print(f"Error adding news overlay: {e}")
        return frame


def _get_status_bar_height():
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    sample_time = datetime.now().strftime("%B %d %Y, %I:%M %p")
    sample_weather = "Temp: 99.9 F, Rain predicted"
    text_size_time, _ = cv2.getTextSize(sample_time, font, font_scale, thickness)
    text_size_weather, _ = cv2.getTextSize(sample_weather, font, font_scale, thickness)
    return max(text_size_time[1], text_size_weather[1]) + 20


def add_time_overlay(frame, temp, weather, status_text=None):
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None

        time_text = datetime.now().strftime("%B %d %Y, %I:%M %p")
        if temp is not None:
            weather_text = f"Temp: {temp:.1f} F"
        else:
            weather_text = "Weather data unavailable"

        if weather:
            w = weather.lower()
            if "rain" in w:
                if "drizzle" in w:
                    weather_text += ", Raining"
                else:
                    weather_text += ", Rain predicted"
            elif "snow" in w:
                if "flurries" in w:
                    weather_text += ", Snowing"
                else:
                    weather_text += ", Snow predicted"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 1

        text_size_time, _ = cv2.getTextSize(time_text, font, font_scale, thickness)
        text_size_weather, _ = cv2.getTextSize(weather_text, font, font_scale, thickness)

        text_x_weather = 10
        text_y = frame.shape[0] - 12
        text_x_time = frame.shape[1] - text_size_time[0] - 10

        overlay_frame = frame.copy()
        bar_height = _get_status_bar_height()
        overlay = overlay_frame.copy()
        cv2.rectangle(
            overlay,
            (0, frame.shape[0] - bar_height),
            (frame.shape[1], frame.shape[0]),
            (50, 50, 50),
            -1,
        )
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)

        cv2.putText(
            overlay_frame,
            weather_text,
            (text_x_weather, text_y),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay_frame,
            time_text,
            (text_x_time, text_y),
            font,
            font_scale,
            font_color,
            thickness,
            cv2.LINE_AA,
        )

        if status_text:
            status_font_scale = 0.5
            status_thickness = 1
            status_size, _ = cv2.getTextSize(
                status_text, font, status_font_scale, status_thickness
            )
            status_position = (frame.shape[1] - status_size[0] - 10, 30)
            cv2.putText(
                overlay_frame,
                status_text,
                (status_position[0] + 1, status_position[1] + 1),
                font,
                status_font_scale,
                (0, 0, 0),
                status_thickness,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_frame,
                status_text,
                status_position,
                font,
                status_font_scale,
                font_color,
                status_thickness,
                cv2.LINE_AA,
            )

        return overlay_frame
    except Exception as e:
        print(f"Error adding overlay: {e}")
        return frame


# ---------------------------------------------------------------------------
# Quotes
# ---------------------------------------------------------------------------

_quotes_cache = None


def load_quotes(quotes_file=None):
    global _quotes_cache
    if _quotes_cache is not None:
        return _quotes_cache

    if quotes_file is None:
        quotes_file = resource_path("quotes.json")

    try:
        with open(quotes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            _quotes_cache = data
        else:
            print(f"quotes.json format is unexpected (expected list).")
            _quotes_cache = []
    except FileNotFoundError:
        print(f"No quotes.json found at {quotes_file}. Using fallback quote.")
        _quotes_cache = []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Failed to load quotes from {quotes_file}: {exc}")
        _quotes_cache = []

    return _quotes_cache


def get_random_quote(quotes_file=None):
    """Gets a random quote from a JSON file, with safe fallback."""
    quotes_list = load_quotes(quotes_file)
    if not quotes_list:
        return "Every day is a fresh start.", "Unknown"

    quote_data = random.choice(quotes_list)
    quote = quote_data.get("quote", "") or ""
    source = quote_data.get("author", "") or ""
    return quote, source


def add_quote_overlay(frame, quote, source="", title=None, style=None):
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None
        MIN_BOX_WIDTH = 600

        quote = sanitize_text(quote)
        if title:
            title = sanitize_text(title)
        source = sanitize_text(source) if source else ""
        style = sanitize_text(style) if style else ""

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_quote = 0.8
        font_scale_source = 0.6
        font_scale_title = 0.9

        font_color = (105, 105, 105)
        font_color_title = (230, 230, 230)
        box_color = (255, 255, 255)
        title_bar_color = (50, 50, 50)
        thickness = 1

        raw_quote_lines = quote.split("\n")
        quote_lines = []
        for raw_line in raw_quote_lines:
            if raw_line.strip():
                wrapped = textwrap.wrap(raw_line.strip(), width=50)
            else:
                wrapped = [""]
            quote_lines.extend(wrapped)

        title_lines = textwrap.wrap(title, width=50) if title else []

        if source.strip().lower() == "today's weather":
            source_lines = []
        else:
            source_lines = textwrap.wrap(f"- {source}", width=50) if source else []

        line_height_quote = cv2.getTextSize(
            "Test", font, font_scale_quote * 1.3, thickness
        )[0][1]
        line_height_title = cv2.getTextSize(
            "Test", font, font_scale_title * 1.3, thickness
        )[0][1]
        line_height_source = (
            cv2.getTextSize("Test", font, font_scale_source * 1.3, thickness)[0][1]
            if source_lines
            else 0
        )

        text_height = 0
        title_height = 0
        if title_lines:
            title_height = line_height_title * len(title_lines) + 20
            text_height += title_height

        text_height += line_height_quote * len(quote_lines)
        if source_lines:
            text_height += 20 + (line_height_source * len(source_lines))

        max_line_widths = []
        if title_lines:
            for line in title_lines:
                max_line_widths.append(
                    cv2.getTextSize(line, font, font_scale_title, thickness)[0][0]
                )
        for line in quote_lines:
            if line:
                max_line_widths.append(
                    cv2.getTextSize(line, font, font_scale_quote, thickness)[0][0]
                )
        if source_lines:
            for line in source_lines:
                max_line_widths.append(
                    cv2.getTextSize(line, font, font_scale_source, thickness)[0][0]
                )

        calculated_width = max(max_line_widths) if max_line_widths else 200
        box_width = max(calculated_width + 40, MIN_BOX_WIDTH)
        box_height = text_height + 80
        box_x = (frame.shape[1] - box_width) // 2
        box_y = (frame.shape[0] - box_height) // 2

        overlay_frame = frame.copy()
        cv2.rectangle(
            overlay_frame,
            (box_x, box_y),
            (box_x + box_width, box_y + box_height),
            box_color,
            -1,
        )
        alpha = 0.8
        cv2.addWeighted(overlay_frame, alpha, frame, 1 - alpha, 0, overlay_frame)

        if title_lines:
            title_bar_y_end = box_y + title_height
            cv2.rectangle(
                overlay_frame,
                (box_x, box_y),
                (box_x + box_width, title_bar_y_end),
                title_bar_color,
                -1,
            )

        title_line_sizes = []
        for line in title_lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale_title, thickness)
            title_line_sizes.append(text_size)

        if title_lines:
            total_title_text_height = sum(t[1] for t in title_line_sizes) + (
                (len(title_lines) - 1) * 10
            )
            available_space = title_height - 20
            vertical_offset = (available_space - total_title_text_height) // 2
            y = box_y + 10 + vertical_offset
        else:
            y = box_y + 20

        for i, line in enumerate(title_lines):
            text_size = title_line_sizes[i]
            line_height = text_size[1]
            x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(
                overlay_frame,
                line,
                (x, y + line_height),
                font,
                font_scale_title,
                font_color_title,
                thickness,
                cv2.LINE_AA,
            )
            y += line_height
            if i < len(title_lines) - 1:
                y += 10

        if title_lines:
            y = box_y + title_height
        y += 20

        for line in quote_lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale_quote, thickness)
            x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(
                overlay_frame,
                line,
                (x, y + text_size[1]),
                font,
                font_scale_quote,
                font_color,
                thickness,
                cv2.LINE_AA,
            )
            y += text_size[1] + 10

        if source_lines:
            y += 10
            for line in source_lines:
                text_size, _ = cv2.getTextSize(line, font, font_scale_source, thickness)
                x = (frame.shape[1] - text_size[0]) // 2
                cv2.putText(
                    overlay_frame,
                    line,
                    (x, y + text_size[1]),
                    font,
                    font_scale_source,
                    font_color,
                    thickness,
                    cv2.LINE_AA,
                )
                y += text_size[1] + 10

        return overlay_frame
    except Exception as e:
        print(f"Error adding quote overlay: {e}")
        return frame


# ---------------------------------------------------------------------------
# Video helpers
# ---------------------------------------------------------------------------

def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"Could not read first frame of video {video_path}")
        return None
    frame = resize_and_pad(frame, frame_width, frame_height)
    return frame


def play_video(
    video_path,
    temp,
    weather,
    status_text=None,
    ambient_dark=False,
    ui_state=None,
    stop_check=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return get_first_frame(video_path), False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30
    wait = int(1000 / fps)

    last_frame = None
    quit_requested = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_and_pad(frame, frame_width, frame_height)
        if frame is None:
            continue
        last_frame = frame.copy()
        present_frame(
            frame,
            temp,
            weather,
            status_text=status_text,
            ambient_dark=ambient_dark,
            ui_state=ui_state,
        )
        key = cv2.waitKey(wait)
        if key == ord("q"):
            quit_requested = True
            break
        if stop_check and stop_check():
            break

    cap.release()

    if last_frame is None:
        last_frame = get_first_frame(video_path)

    return last_frame, quit_requested


# ---------------------------------------------------------------------------
# Image loading cache
# ---------------------------------------------------------------------------

_image_cache = {}
_MAX_IMAGE_CACHE = 10


def load_scaled_image(path):
    """Load an image from disk, downscale it, and cache a limited number."""
    global _image_cache

    if path in _image_cache:
        return _image_cache[path]

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to read image: {path}")
        return None

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] > 4:
            img = img[:, :, :4]

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    max_dim = max(frame_width, frame_height) * 2  # e.g. 1600 for 800x480
    scale = min(max_dim / float(w), max_dim / float(h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(_image_cache) >= _MAX_IMAGE_CACHE:
        # Evict an arbitrary entry (simple but effective)
        try:
            _image_cache.pop(next(iter(_image_cache)))
        except StopIteration:
            pass

    _image_cache[path] = img
    return img

# ---------------------------------------------------------------------------
# Ambient brightness / camera helpers
# ---------------------------------------------------------------------------

_camera = None  # global camera handle for brightness sampling

def get_ambient_brightness():
    """
    Capture a single frame from the default camera and return its average
    grayscale brightness (0-255). Returns None on error.
    """
    global _camera

    if _camera is None:
        _camera = cv2.VideoCapture(1)  # change index if needed (e.g., 1)
        if not _camera.isOpened():
            print("Unable to open camera for brightness detection.")
            _camera.release()
            _camera = None
            return None

    ret, frame = _camera.read()
    if not ret or frame is None:
        print("Failed to capture frame from camera for brightness detection.")
        return None

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(gray.mean())
    except Exception as exc:
        print(f"Error computing ambient brightness: {exc}")
        return None

def release_camera():
    """Release the ambient brightness camera, if open."""
    global _camera
    if _camera is not None:
        _camera.release()
        _camera = None


def build_mode_buttons():
    count = len(MODE_DEFINITIONS)
    margin = 10
    gap = 8
    height = 44
    y_pos = 8
    total_width = frame_width - (margin * 2) - (gap * (count - 1))
    button_width = max(80, int(total_width / count))

    buttons = []
    x_pos = margin
    for entry in MODE_DEFINITIONS:
        rect = (x_pos, y_pos, x_pos + button_width, y_pos + height)
        buttons.append({"mode": entry["mode"], "label": entry["label"], "rect": rect})
        x_pos += button_width + gap
    return buttons


def draw_mode_buttons(frame, buttons, active_mode):
    if not buttons:
        return frame

    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for button in buttons:
        x1, y1, x2, y2 = button["rect"]
        is_active = button["mode"] == active_mode
        bg_color = (0, 130, 210) if is_active else (60, 60, 60)
        border_color = (255, 255, 255)
        text_color = (255, 255, 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 1)

        label = button["label"]
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            font,
            font_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

    return overlay


# ---------------------------------------------------------------------------
# Display frame builder
# ---------------------------------------------------------------------------

def build_display_frame(media_item, image_paths, forecast_5day):
    """
    Build a display frame for an image media item.

    Chooses between "single", "stitch", "quote", "forecast", "today", "news"
    depending on time of day and day of week.
    """
    if media_item["type"] != "image":
        return None

    base_img = load_scaled_image(media_item["path"])
    if base_img is None:
        return None

    ct = get_central_time()
    current_hour = ct.hour
    current_day = ct.weekday()  # Monday=0

    if 7 <= current_hour < 18:
        if current_day < 5:
            valid_display_types = ["single", "stitch", "quote", "forecast", "today", "news"]
            weights = [2, 2, 2, 2, 2, 6]
        else:
            valid_display_types = ["single", "stitch", "quote", "forecast", "today", "news"]
            weights = [3, 2, 2, 2, 2, 3]
    else:
        valid_display_types = ["single", "quote", "today"]
        weights = [3, 2, 2]

    display_type = random.choices(valid_display_types, weights=weights, k=1)[0]

    if display_type == "forecast":
        frame = create_zoomed_blurred_background(base_img, frame_width, frame_height)
        frame = add_forecast_overlay(frame, forecast_5day)
        return frame

    if display_type == "stitch":
        stitch_count = random.randint(2, 4)
        other_paths = [p for p in image_paths if p != media_item["path"]]
        pool_paths = []
        if len(other_paths) >= stitch_count:
            pool_paths = random.sample(other_paths, stitch_count)
        elif other_paths:
            pool_paths = other_paths
        else:
            pool_paths = [media_item["path"]]

        images = []
        for p in pool_paths:
            img = load_scaled_image(p)
            if img is not None:
                images.append(img)
        if not images:
            return create_single_image_with_background(base_img, frame_width, frame_height)
        stitched = stitch_images(images, frame_width, frame_height)
        return stitched

    if display_type == "quote":
        frame = create_zoomed_blurred_background(base_img, frame_width, frame_height)
        quote, source = get_random_quote()
        frame = add_quote_overlay(frame, quote, source)
        return frame

    if display_type == "today":
        frame = create_zoomed_blurred_background(base_img, frame_width, frame_height)
        city = "Waupun"
        country_code = "US"
        weather_data = get_weather_forecast2(api_key, city, country_code)

        forecast_summary = "Weather summary unavailable."
        custom_title = "Today's Forecast"

        if weather_data:
            forecast_summary, style_used = get_or_generate_forecast_summary(weather_data, style="random")
            custom_title = style_titles.get(style_used, "Today's Forecast")
        else:
            print("No weather data available for daily forecast summary.")

        frame = add_quote_overlay(
            frame,
            quote=forecast_summary,
            source="Today's Weather",
            title=custom_title,
            style=None,
        )
        return frame

    if display_type == "news":
        frame = create_zoomed_blurred_background(base_img, frame_width, frame_height)
        news = get_ai_generated_news()
        frame = add_news_overlay(frame, news)
        return frame

    # Fallback / default
    frame = create_single_image_with_background(base_img, frame_width, frame_height)
    return frame


# ---------------------------------------------------------------------------
# Slideshow core
# ---------------------------------------------------------------------------

def build_display_frame_for_mode(media_item, image_paths, forecast_5day, mode):
    if media_item["type"] != "image":
        return None

    if mode == "random":
        return build_display_frame(media_item, image_paths, forecast_5day)

    base_img = load_scaled_image(media_item["path"])
    if base_img is None:
        return None

    if mode == "news":
        frame = create_zoomed_blurred_background(base_img, frame_width, frame_height)
        news = get_ai_generated_news()
        return add_news_overlay(frame, news)

    if mode == "weather":
        frame = create_zoomed_blurred_background(base_img, frame_width, frame_height)
        if forecast_5day and random.random() < 0.5:
            return add_forecast_overlay(frame, forecast_5day)

        city = "Waupun"
        country_code = "US"
        weather_data = get_weather_forecast2(api_key, city, country_code)
        forecast_summary = "Weather summary unavailable."
        custom_title = "Today's Forecast"

        if weather_data:
            forecast_summary, style_used = get_or_generate_forecast_summary(
                weather_data, style="random"
            )
            custom_title = style_titles.get(style_used, "Today's Forecast")

        return add_quote_overlay(
            frame,
            quote=forecast_summary,
            source="Today's Weather",
            title=custom_title,
            style=None,
        )

    if mode == "pictures":
        return create_single_image_with_background(base_img, frame_width, frame_height)

    return build_display_frame(media_item, image_paths, forecast_5day)

def run_slideshow_once():
    """
    Run one full slideshow session.

    Returns True if the user requested exit (pressing 'q'),
    otherwise False (to allow restart on error).
    """
    # NOTE: Replace this with your Drive folder ID
    folder_id = "1hpBzZ_kiXpIBtRv1FN3da8zOhT5J0Ggi"

    temp_dir = resource_path("images")
    metadata_file = resource_path("metadata.json")

    service = authenticate_drive()
    local_metadata = load_local_metadata(metadata_file)
    downloaded_files = set()

    if service is not None:
        media_items, local_metadata, downloaded_files = refresh_media_items(
            service, folder_id, temp_dir, metadata_file, local_metadata
        )
        if media_items is None:
            print("Unable to retrieve media from Google Drive. Using cached media if available.")
            media_items = load_media_from_local_cache(temp_dir)
    else:
        print("Unable to authenticate with Google Drive. Using cached media if available.")
        media_items = load_media_from_local_cache(temp_dir)

    if not media_items:
        print("No media available to display.")
        return False

    image_paths = [item["path"] for item in media_items if item["type"] == "image"]
    if not image_paths:
        print("Warning: no images found; slideshow will play videos only.")

    forecast_5day = get_weather_forecast(api_key) or []

    basic_transitions = [
        fade_transition,
        slide_transition_left,
        slide_transition_right,
        wipe_transition_top,
        wipe_transition_bottom,
    ]
    advanced_transitions = [
        melt_transition,
        wave_transition,
        zen_ripple_transition,
        dynamic_petal_bloom_transition,
    ]

    if LOW_POWER_MODE:
        transitions = [fade_transition]
        global _LOW_POWER_NOTICE_SHOWN
        if not _LOW_POWER_NOTICE_SHOWN:
            print("Low-power mode enabled: using simplified transitions.")
            _LOW_POWER_NOTICE_SHOWN = True
    else:
        transitions = basic_transitions + advanced_transitions

    cv2.namedWindow("slideshow", cv2.WINDOW_NORMAL)
    ensure_fullscreen("slideshow")
    hide_mouse_cursor()

    buttons = build_mode_buttons()
    mode_state = {
        "mode": "random",
        "dirty": True,
        "fallback": False,
        "buttons": buttons,
        "show_buttons": False,
        "last_touch": None,
        "needs_redraw": False,
    }

    def handle_touch(event, x, y, flags, params):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        now = time.monotonic()
        if not mode_state["show_buttons"]:
            mode_state["show_buttons"] = True
            mode_state["last_touch"] = now
            mode_state["needs_redraw"] = True
            return
        mode_state["last_touch"] = now
        mode_state["needs_redraw"] = True
        for button in buttons:
            x1, y1, x2, y2 = button["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                new_mode = button["mode"]
                if new_mode != mode_state["mode"]:
                    mode_state["mode"] = new_mode
                    mode_state["dirty"] = True
                return

    cv2.setMouseCallback("slideshow", handle_touch)

    index = 0
    current_item = media_items[index]
    play_queue = []
    active_items = media_items[:]
    active_image_paths = [item["path"] for item in active_items if item["type"] == "image"]
    last_refresh_time = datetime.now()
    exit_requested = False
    ambient_dark = False
    last_brightness_check = datetime.min

    # This holds the already-built frame for the current_item (for images).
    current_frame = None

    def get_next_index(current_idx):
        nonlocal play_queue, active_items
        total = len(active_items)
        if total <= 1:
            return current_idx

        # prune invalid indices and current index
        play_queue = [i for i in play_queue if 0 <= i < total and i != current_idx]
        if not play_queue:
            play_queue = [i for i in range(total) if i != current_idx]
            random.shuffle(play_queue)
        return play_queue.pop(0)

    def get_mode_label(mode):
        for entry in MODE_DEFINITIONS:
            if entry["mode"] == mode:
                return entry["label"]
        return mode.title()

    def rebuild_active_items(reason):
        nonlocal active_items, active_image_paths, index, current_item, current_frame, play_queue
        if mode_state["mode"] == "video":
            filtered = [item for item in media_items if item["type"] == "video"]
        elif mode_state["mode"] in {"news", "weather", "pictures"}:
            filtered = [item for item in media_items if item["type"] == "image"]
        else:
            filtered = list(media_items)

        if not filtered:
            active_items = list(media_items)
            mode_state["fallback"] = True
        else:
            active_items = filtered
            mode_state["fallback"] = False

        active_image_paths = [item["path"] for item in active_items if item["type"] == "image"]
        index = 0
        current_item = active_items[index]
        current_frame = None
        play_queue = []
        mode_state["dirty"] = False
        if reason:
            print(reason)

    try:
        current_frame = None
        rebuild_active_items("Touch mode ready: Random.")

        while True:
            now = datetime.now()
            if mode_state["dirty"]:
                rebuild_active_items(f"Mode switched to {get_mode_label(mode_state['mode'])}.")

            # ---------------------------------------------------------------
            # Ambient brightness / lights-on detection
            # ---------------------------------------------------------------
            if now - last_brightness_check >= BRIGHTNESS_CHECK_INTERVAL:
                b = get_ambient_brightness()
                last_brightness_check = now
                if b is not None:
                    ambient_dark = b < BRIGHTNESS_DARK_THRESHOLD
                    # Optional debug:
                    # print(f"Ambient brightness: {b:.1f} -> dark={ambient_dark}")

            # Periodic Drive refresh
            if service is not None and now - last_refresh_time >= MEDIA_REFRESH_INTERVAL:
                refreshed_media, new_metadata, new_downloaded = refresh_media_items(
                    service, folder_id, temp_dir, metadata_file, local_metadata
                )
                last_refresh_time = now
                if refreshed_media is not None and refreshed_media:
                    media_items = refreshed_media
                    local_metadata = new_metadata
                    downloaded_files = new_downloaded
                    mode_state["dirty"] = True
                    print("Media playlist refreshed.")
                elif refreshed_media == []:
                    print("Drive folder is empty after refresh; keeping existing playlist.")
                else:
                    print("Media refresh failed; keeping existing playlist.")

            # Occasionally refresh 5-day forecast (e.g., on the hour)
            if now.minute == 0 and now.second < 5:
                refreshed = get_weather_forecast(api_key)
                if refreshed is not None:
                    forecast_5day = refreshed

            temp, weather = get_weather_data(api_key)
            status_text = f"Mode: {get_mode_label(mode_state['mode'])}"
            if mode_state["fallback"]:
                status_text += " (fallback: no items)"

            # --- Display current item ---
            try:
                if current_item["type"] == "video":
                    current_frame, quit_requested = play_video(
                        current_item["path"],
                        temp,
                        weather,
                        status_text=status_text,
                        ambient_dark=ambient_dark,
                        ui_state=mode_state,
                        stop_check=lambda: mode_state["dirty"],
                    )
                    if quit_requested:
                        exit_requested = True
                        break
                    if mode_state["dirty"]:
                        continue
                    if current_frame is None:
                        current_frame = get_first_frame(current_item["path"])
                        if current_frame is None:
                            print(
                                f"Video {current_item['name']} could not be decoded; skipping."
                            )
                else:
                    # For images, build the frame only once per "visit"
                    if current_frame is None:
                        frame = build_display_frame_for_mode(
                            current_item, active_image_paths, forecast_5day, mode_state["mode"]
                        )
                        if frame is None:
                            print(
                                f"Image {current_item['name']} could not be loaded; skipping."
                            )
                            if len(active_items) > 1:
                                media_items = [
                                    item for item in media_items if item["name"] != current_item["name"]
                                ]
                                mode_state["dirty"] = True
                                continue
                            else:
                                print("No valid media left to display.")
                                return False
                        current_frame = frame

                    present_frame(
                        current_frame,
                        temp,
                        weather,
                        status_text=status_text,
                        ambient_dark=ambient_dark,
                        ui_state=mode_state,
                    )
                    wait_start = time.time()
                    while time.time() - wait_start < display_time:
                        if mode_state["show_buttons"] or mode_state["needs_redraw"]:
                            present_frame(
                                current_frame,
                                temp,
                                weather,
                                status_text=status_text,
                                ambient_dark=ambient_dark,
                                ui_state=mode_state,
                            )
                            mode_state["needs_redraw"] = False
                        key = cv2.waitKey(100)
                        if key == ord("q"):
                            exit_requested = True
                            break
                        if mode_state["dirty"]:
                            break
                    if exit_requested or mode_state["dirty"]:
                        continue
            except KeyboardInterrupt:
                exit_requested = True
                break
            except Exception as exc:
                print(
                    f"Error displaying item {current_item.get('name', '<unknown>')}: {exc}"
                )
                traceback.print_exc()
                if len(active_items) <= 1:
                    return False
                index = (index + 1) % len(active_items)
                current_item = active_items[index]
                continue

            if exit_requested:
                break

            # --- Pick and prepare next item for transition ---
            if len(active_items) > 1:
                next_index = get_next_index(index)
                next_item = active_items[next_index]

                next_frame = None
                if next_item["type"] == "video":
                    next_frame = get_first_frame(next_item["path"])
                else:
                    next_frame = build_display_frame_for_mode(
                        next_item, active_image_paths, forecast_5day, mode_state["mode"]
                    )

                if next_frame is None:
                    print(
                        f"Failed to prepare next frame for {next_item['name']}. Skipping transition."
                    )
                else:
                    # Make sure we have a valid starting frame
                    if current_frame is None:
                        current_frame = next_frame

                    transition_fn = random.choice(transitions)
                    for frame in transition_fn(current_frame, next_frame, num_transition_frames):
                        present_frame(
                            frame,
                            temp,
                            weather,
                            status_text=status_text,
                            ambient_dark=ambient_dark,
                            ui_state=mode_state,
                        )
                        if cv2.waitKey(1) == ord("q"):
                            exit_requested = True
                            break
                        if mode_state["dirty"]:
                            break
                    if exit_requested:
                        break
                    if mode_state["dirty"]:
                        continue

                    # The frame we faded into becomes the new "current_frame"
                    current_frame = next_frame

                current_item = next_item
                index = next_index
            else:
                # Only one media item, just loop it
                continue

    finally:
        release_camera()
        show_mouse_cursor()
        cv2.destroyAllWindows()

    return exit_requested


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    consecutive_failures = 0
    base_delay = 5  # seconds

    while True:
        try:
            exit_requested = run_slideshow_once()
            if exit_requested:
                print("Slideshow exited by user.")
                break
            consecutive_failures = 0
        except KeyboardInterrupt:
            print("Received interrupt. Shutting down slideshow.")
            break
        except Exception as exc:
            consecutive_failures += 1
            print(f"Fatal error in slideshow: {exc}")
            traceback.print_exc()

        wait_seconds = (
            base_delay
            if consecutive_failures == 0
            else min(60, base_delay * (consecutive_failures + 1))
        )
        print(f"Restarting slideshow in {wait_seconds} seconds...")
        time.sleep(wait_seconds)


if __name__ == "__main__":
    main()


