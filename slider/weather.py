"""Weather data from OpenWeatherMap: current conditions, forecasts, and icons.

Network calls are cached on disk (TTL from config) and always degrade to the
last good cached value, so callers never see an exception from this module.
"""

import math
import os

import cv2
import numpy as np
import requests
from requests import RequestException

from slider import config
from slider.utils import from_timestamp, now_local, read_timed_cache, write_timed_cache

BASE_URL = "https://api.openweathermap.org/data/2.5"


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def _fetch(endpoint, city):
    """GET an OpenWeatherMap endpoint. Returns the parsed dict or None."""
    api_key = config.WEATHERMAP_API_KEY
    if not api_key:
        return None
    params = {
        "q": f"{city},{config.WEATHER_COUNTRY_CODE}",
        "units": "imperial",
        "appid": api_key,
    }
    try:
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except (RequestException, ValueError) as exc:
        print(f"Error fetching weather ({endpoint}): {exc}")
        return None
    return data if isinstance(data, dict) else None


def _num(value, default=None):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


# ---------------------------------------------------------------------------
# Current conditions
# ---------------------------------------------------------------------------

def _parse_current(data):
    try:
        main = data["main"]
        weather = data["weather"][0]
        temp = _num(main.get("temp"))
        if temp is None:
            return None
        sys_info = data.get("sys") or {}
        wind = data.get("wind") or {}
        return {
            "temp": temp,
            "feels_like": _num(main.get("feels_like"), temp),
            "temp_min": _num(main.get("temp_min"), temp),
            "temp_max": _num(main.get("temp_max"), temp),
            "humidity": _num(main.get("humidity"), 0),
            "wind_speed": _num(wind.get("speed"), 0),
            "wind_gust": _num(wind.get("gust"), 0),
            "main": str(weather.get("main") or ""),
            "description": str(weather.get("description") or weather.get("main") or ""),
            "sunrise": _num(sys_info.get("sunrise")),
            "sunset": _num(sys_info.get("sunset")),
            "city": str(data.get("name") or config.WEATHER_CURRENT_CITY),
        }
    except (KeyError, IndexError, TypeError, AttributeError):
        return None


def get_current_weather():
    """Return a dict of current conditions, or None if nothing is available.

    Keys: temp, feels_like, temp_min, temp_max, humidity, wind_speed,
    wind_gust, main, description, sunrise, sunset, city.
    """
    cached, fresh = read_timed_cache(config.WEATHER_CACHE_FILE, config.WEATHER_CACHE_TTL)
    cached_value = cached.get("current") if cached else None
    if fresh and isinstance(cached_value, dict):
        return cached_value

    if not config.WEATHERMAP_API_KEY:
        if cached is None:
            print("OpenWeatherMap API key is missing.")
        return cached_value if isinstance(cached_value, dict) else None

    data = _fetch("weather", config.WEATHER_CURRENT_CITY)
    current = _parse_current(data) if data else None
    if current:
        write_timed_cache(config.WEATHER_CACHE_FILE, {"current": current})
        return current
    return cached_value if isinstance(cached_value, dict) else None


# ---------------------------------------------------------------------------
# 3-hour forecast entries (shared by today's forecast and the 5-day view)
# ---------------------------------------------------------------------------

def _simplify_entries(data):
    entries = []
    for item in data.get("list", []) or []:
        try:
            main = item["main"]
            weather = item["weather"][0]
            dt = int(item["dt"])
            temp = _num(main.get("temp"))
            if temp is None:
                continue
            wind = item.get("wind") or {}
            entries.append({
                "dt": dt,
                "temp": temp,
                "feels_like": _num(main.get("feels_like"), temp),
                "temp_min": _num(main.get("temp_min"), temp),
                "temp_max": _num(main.get("temp_max"), temp),
                "humidity": _num(main.get("humidity"), 0),
                "wind_speed": _num(wind.get("speed"), 0),
                "wind_gust": _num(wind.get("gust"), 0),
                "pop": _num(item.get("pop"), 0),
                "rain": _num((item.get("rain") or {}).get("3h"), 0),
                "snow": _num((item.get("snow") or {}).get("3h"), 0),
                "main": str(weather.get("main") or ""),
                "description": str(weather.get("description") or weather.get("main") or ""),
            })
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    entries.sort(key=lambda e: e["dt"])
    return entries


def get_forecast_entries():
    """Return the cached list of 3-hour forecast entries (may be empty)."""
    cached, fresh = read_timed_cache(config.FORECAST_CACHE_FILE, config.FORECAST_CACHE_TTL)
    cached_entries = cached.get("entries") if cached else None
    if fresh and isinstance(cached_entries, list) and cached_entries:
        return cached_entries

    if not config.WEATHERMAP_API_KEY:
        if cached is None:
            print("OpenWeatherMap API key is missing.")
        return cached_entries if isinstance(cached_entries, list) else []

    data = _fetch("forecast", config.WEATHER_FORECAST_CITY)
    entries = _simplify_entries(data) if data else []
    if entries:
        write_timed_cache(config.FORECAST_CACHE_FILE, {"entries": entries})
        return entries
    return cached_entries if isinstance(cached_entries, list) else []


def _slice_view(entry):
    local = from_timestamp(entry["dt"])
    return {
        "time": local.strftime("%I:%M %p").lstrip("0"),
        "temp": entry["temp"],
        "feels_like": entry.get("feels_like", entry["temp"]),
        "description": entry["description"],
        "wind_speed": entry.get("wind_speed", 0),
        "humidity": entry.get("humidity", 0),
        "pop": entry.get("pop", 0),
    }


def get_todays_forecast():
    """Return the remaining 3-hour slices for today.

    Late in the evening, when fewer than two slices remain, the next 24 hours
    are returned instead so the AI summary always has something to describe.
    """
    entries = get_forecast_entries()
    if not entries:
        return []
    now = now_local()
    cutoff = now.timestamp() - 3 * 3600  # keep the slice we are currently in
    upcoming = [e for e in entries if e["dt"] >= cutoff]
    todays = [e for e in upcoming if from_timestamp(e["dt"]).date() == now.date()]
    chosen = todays if len(todays) >= 2 else upcoming[:8]
    return [_slice_view(e) for e in chosen]


def get_5day_forecast():
    """Return up to five daily summaries, or None when no data is available.

    Each entry: date ("Monday, Sep 23"), date_key (ISO), temp_min, temp_max,
    description, pop (0-1 chance of precipitation).
    """
    entries = get_forecast_entries()
    if not entries:
        return None

    days = {}
    for entry in entries:
        local = from_timestamp(entry["dt"])
        key = local.date()
        day = days.setdefault(key, {
            "temp_min": math.inf, "temp_max": -math.inf,
            "descriptions": [], "pop": 0.0, "noon": None,
        })
        day["temp_min"] = min(day["temp_min"], entry.get("temp_min", entry["temp"]))
        day["temp_max"] = max(day["temp_max"], entry.get("temp_max", entry["temp"]))
        day["descriptions"].append(entry["description"])
        day["pop"] = max(day["pop"], entry.get("pop", 0) or 0)
        if local.hour == 12:
            day["noon"] = entry["description"]

    forecast = []
    for key in sorted(days):
        day = days[key]
        descriptions = day["descriptions"]
        dominant = day["noon"] or max(set(descriptions), key=descriptions.count)
        forecast.append({
            "date": key.strftime("%A, %b %d"),
            "date_key": key.isoformat(),
            "temp_min": day["temp_min"],
            "temp_max": day["temp_max"],
            "description": dominant,
            "pop": day["pop"],
        })
    return forecast[:5]


def format_clock(unix_ts):
    """Format a Unix timestamp as e.g. '6:42 AM' in the configured timezone."""
    if unix_ts is None:
        return ""
    return from_timestamp(unix_ts).strftime("%I:%M %p").lstrip("0")


def build_ai_context(current, today, five_day):
    """Bundle the pieces the AI summary needs. Any argument may be None/empty."""
    return {
        "city": (current or {}).get("city") or config.WEATHER_FORECAST_CITY,
        "current": current or None,
        "today": list(today or []),
        "five_day": list(five_day or []),
    }


# ---------------------------------------------------------------------------
# Weather icons: PNGs from icons/ when present, otherwise drawn procedurally
# ---------------------------------------------------------------------------

_ICON_FILES = {
    "clear": "sunny.png",
    "partly": "cloudy.png",
    "cloudy": "cloudy.png",
    "rain": "rain.png",
    "storm": "rain.png",
    "snow": "snow.png",
    "windy": "windy.png",
    "fog": "cloudy.png",
}
_icon_cache = {}


def icon_kind(description):
    """Map a weather description to an icon key, or None."""
    d = (description or "").lower()
    if "thunder" in d or "storm" in d:
        return "storm"
    if "snow" in d or "sleet" in d or "flurr" in d:
        return "snow"
    if "rain" in d or "drizzle" in d or "shower" in d:
        return "rain"
    if "wind" in d or "breez" in d or "gale" in d:
        return "windy"
    if any(k in d for k in ("mist", "fog", "haze", "smoke", "dust", "sand", "ash")):
        return "fog"
    if "clear" in d or "sunny" in d:
        return "clear"
    if "few clouds" in d or "scattered" in d or "partly" in d:
        return "partly"
    if "cloud" in d or "overcast" in d:
        return "cloudy"
    return None


_SUN = (0, 200, 255, 255)
_CLOUD = (235, 235, 235, 255)
_CLOUD_DARK = (150, 150, 150, 255)
_RAIN = (255, 170, 60, 255)
_SNOW = (255, 245, 235, 255)
_WIND = (200, 190, 160, 255)
_FOG = (205, 205, 205, 255)


def _cloud(canvas, cx, cy, s, color):
    cv2.circle(canvas, (int(cx - 0.35 * s), int(cy + 0.05 * s)), int(0.30 * s), color, -1)
    cv2.circle(canvas, (int(cx), int(cy - 0.15 * s)), int(0.42 * s), color, -1)
    cv2.circle(canvas, (int(cx + 0.38 * s), int(cy + 0.05 * s)), int(0.30 * s), color, -1)
    cv2.rectangle(canvas, (int(cx - 0.35 * s), int(cy + 0.05 * s)), (int(cx + 0.38 * s), int(cy + 0.35 * s)), color, -1)


def _sun(canvas, cx, cy, r):
    cv2.circle(canvas, (cx, cy), r, _SUN, -1)
    for i in range(8):
        angle = i * math.pi / 4
        x1 = int(cx + (r + 14) * math.cos(angle))
        y1 = int(cy + (r + 14) * math.sin(angle))
        x2 = int(cx + (r + 40) * math.cos(angle))
        y2 = int(cy + (r + 40) * math.sin(angle))
        cv2.line(canvas, (x1, y1), (x2, y2), _SUN, 12)


def _draw_icon(kind, size):
    big = 256
    canvas = np.zeros((big, big, 4), dtype=np.uint8)
    if kind == "clear":
        _sun(canvas, 128, 128, 52)
    elif kind == "partly":
        _sun(canvas, 100, 100, 44)
        _cloud(canvas, 150, 150, 100, _CLOUD)
    elif kind == "cloudy":
        _cloud(canvas, 90, 105, 80, _CLOUD_DARK)
        _cloud(canvas, 140, 140, 105, _CLOUD)
    elif kind == "rain":
        _cloud(canvas, 128, 100, 100, _CLOUD)
        for x in (92, 130, 168):
            cv2.line(canvas, (x, 172), (x - 14, 226), _RAIN, 12)
    elif kind == "snow":
        _cloud(canvas, 128, 100, 100, _CLOUD)
        for x, y in ((92, 195), (130, 215), (168, 195)):
            cv2.circle(canvas, (x, y), 11, _SNOW, -1)
    elif kind == "storm":
        _cloud(canvas, 128, 95, 100, _CLOUD_DARK)
        bolt = np.array([[140, 150], [104, 210], [132, 210], [116, 252], [166, 190], [140, 190], [156, 150]], np.int32)
        cv2.fillPoly(canvas, [bolt], _SUN)
    elif kind == "windy":
        for (x1, x2, y) in ((40, 190, 95), (60, 215, 135), (40, 170, 175)):
            cv2.line(canvas, (x1, y), (x2, y), _WIND, 14)
            cv2.circle(canvas, (x2, y - 16), 16, _WIND, 10)
    elif kind == "fog":
        _cloud(canvas, 128, 90, 85, _CLOUD)
        for y in (165, 195, 225):
            cv2.line(canvas, (52, y), (204, y), _FOG, 12)
    else:
        return None
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def get_weather_icon(description, size=64):
    """Return a BGRA icon for the description (cached), or None if unknown."""
    kind = icon_kind(description)
    if kind is None:
        return None
    key = (kind, size)
    if key in _icon_cache:
        return _icon_cache[key]

    icon = None
    custom = os.path.join(config.ICONS_DIR, _ICON_FILES.get(kind, ""))
    if os.path.isfile(custom):
        loaded = cv2.imread(custom, cv2.IMREAD_UNCHANGED)
        if loaded is not None and loaded.size:
            if loaded.ndim == 2:
                loaded = cv2.cvtColor(loaded, cv2.COLOR_GRAY2BGRA)
            elif loaded.shape[2] == 3:
                loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2BGRA)
            icon = cv2.resize(loaded, (size, size), interpolation=cv2.INTER_AREA)
    if icon is None:
        icon = _draw_icon(kind, size)
    _icon_cache[key] = icon
    return icon
