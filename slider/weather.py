"""Weather data fetching from OpenWeatherMap — current conditions, forecasts, and icons."""

import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

import cv2
import requests
from requests import RequestException

from slider import config

# ---------------------------------------------------------------------------
# Weather icons (lazy-loaded)
# ---------------------------------------------------------------------------

_icon_images = None


def _load_icons():
    global _icon_images
    if _icon_images is not None:
        return _icon_images

    _icon_images = {}
    icon_dir = config.resource_path("icons")
    icon_map = {
        "clear": "sunny.png",
        "cloudy": "cloudy.png",
        "rain": "rain.png",
        "snow": "snow.png",
        "windy": "windy.png",
    }
    for key, filename in icon_map.items():
        path = os.path.join(icon_dir, filename)
        if os.path.exists(path):
            _icon_images[key] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        else:
            _icon_images[key] = None
    return _icon_images


def get_weather_icon(description):
    """Return the appropriate icon image based on the weather description."""
    icons = _load_icons()
    description = description.lower()
    if "clear" in description:
        return icons.get("clear")
    elif "cloud" in description:
        return icons.get("cloudy")
    elif "rain" in description:
        return icons.get("rain")
    elif "wind" in description or "breeze" in description:
        return icons.get("windy")
    elif "snow" in description:
        return icons.get("snow")
    return None


# ---------------------------------------------------------------------------
# Current weather
# ---------------------------------------------------------------------------

def get_current_weather(cache_file=None):
    """Return (temp, weather_description) for the configured city.

    Previously named get_weather_data().
    """
    if cache_file is None:
        cache_file = config.resource_path("weather_cache.json")

    api_key = config.WEATHERMAP_API_KEY
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

    city = config.WEATHER_CURRENT_CITY
    country = config.WEATHER_COUNTRY_CODE
    url = (
        f"http://api.openweathermap.org/data/2.5/weather"
        f"?q={city},{country}&units=imperial&appid={api_key}"
    )
    try:
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
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
                json.dump({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "temp": temp,
                    "weather": weather,
                }, f)
        except OSError as exc:
            print(f"Failed to write weather cache: {exc}")
        return temp, weather

    print("Error fetching weather data")
    return None, None


# ---------------------------------------------------------------------------
# Today's detailed forecast (3-hour slices)
# ---------------------------------------------------------------------------

def get_todays_forecast(cache_file=None):
    """Fetch today's detailed forecast (3h slices) and cache it.

    Previously named get_weather_forecast2().
    """
    if cache_file is None:
        cache_file = config.resource_path("forecast_cache.json")

    api_key = config.WEATHERMAP_API_KEY
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
            if datetime.now() - timestamp < config.FORECAST_CACHE_TTL:
                return cached_forecast
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"Failed to read forecast cache: {exc}")
            cached_forecast = None

    city = config.WEATHER_FORECAST_CITY
    country = config.WEATHER_COUNTRY_CODE
    url = (
        f"http://api.openweathermap.org/data/2.5/forecast"
        f"?q={city},{country}&units=imperial&appid={api_key}"
    )
    try:
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
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
                today_weather.append({
                    "time": dt.strftime("%I:%M %p"),
                    "temp": item["main"]["temp"],
                    "description": item["weather"][0]["description"],
                    "wind_speed": item["wind"]["speed"],
                    "humidity": item["main"]["humidity"],
                })

        try:
            with open(cache_file, "w") as cache_handle:
                json.dump({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "forecast": today_weather,
                }, cache_handle)
        except OSError as exc:
            print(f"Failed to write forecast cache: {exc}")

        return today_weather

    print("Error fetching weather data.")
    return cached_forecast or []


# ---------------------------------------------------------------------------
# 5-day forecast
# ---------------------------------------------------------------------------

def get_5day_forecast():
    """Return a 5-day (min/max) forecast.

    Previously named get_weather_forecast().
    """
    api_key = config.WEATHERMAP_API_KEY
    if not api_key:
        print("OpenWeatherMap API key is missing.")
        return None

    city = config.WEATHER_FORECAST_CITY
    country = config.WEATHER_COUNTRY_CODE
    url = (
        f"http://api.openweathermap.org/data/2.5/forecast"
        f"?q={city},{country}&units=imperial&appid={api_key}"
    )
    try:
        response = requests.get(url, timeout=config.REQUEST_TIMEOUT)
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
            formatted_forecast.append({
                "date": dt.strftime("%A, %b %d"),
                "temp_min": forecast["temp_min"],
                "temp_max": forecast["temp_max"],
                "description": dominant_desc,
            })

        formatted_forecast.sort(key=lambda x: datetime.strptime(x["date"], "%A, %b %d"))
        return formatted_forecast[:5]

    print("Error fetching weather forecast data")
    return None
