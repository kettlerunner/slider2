from datetime import datetime, timedelta

import numpy as np

from slider import config, weather
from slider.utils import local_tz


def _forecast_payload(start, hours, description="light rain", pop=0.4):
    tz = local_tz()
    base = datetime(start.year, start.month, start.day, 0, 0, tzinfo=tz)
    entries = []
    for i in range(0, hours, 3):
        stamp = base + timedelta(hours=i)
        entries.append({
            "dt": int(stamp.timestamp()),
            "main": {"temp": 50 + i % 20, "temp_min": 45 + i % 20, "temp_max": 55 + i % 20,
                     "feels_like": 49, "humidity": 60},
            "weather": [{"main": "Rain", "description": description}],
            "wind": {"speed": 7, "gust": 12},
            "pop": pop,
        })
    return {"list": entries}


def test_5day_sorts_across_year_boundary(isolated_caches, monkeypatch):
    monkeypatch.setattr(config, "WEATHERMAP_API_KEY", "key")
    start = datetime(2027, 12, 30)
    monkeypatch.setattr(weather, "_fetch", lambda endpoint, city: _forecast_payload(start, 24 * 5))
    days = weather.get_5day_forecast()
    assert [d["date_key"] for d in days] == ["2027-12-30", "2027-12-31", "2028-01-01", "2028-01-02", "2028-01-03"]
    assert days[0]["date"].endswith("Dec 30") and days[0]["pop"] == 0.4
    assert days[0]["temp_min"] < days[0]["temp_max"]


def test_forecast_uses_cache_then_stale_on_failure(isolated_caches, monkeypatch):
    monkeypatch.setattr(config, "WEATHERMAP_API_KEY", "key")
    calls = []

    def fake_fetch(endpoint, city):
        calls.append(endpoint)
        return _forecast_payload(datetime.now(), 48)

    monkeypatch.setattr(weather, "_fetch", fake_fetch)
    first = weather.get_forecast_entries()
    second = weather.get_forecast_entries()
    assert first and first == second and calls == ["forecast"]  # second call served from disk

    monkeypatch.setattr(config, "FORECAST_CACHE_TTL", timedelta(seconds=-1))
    monkeypatch.setattr(weather, "_fetch", lambda endpoint, city: None)
    assert weather.get_forecast_entries() == first  # stale cache beats nothing


def test_current_weather_parses_and_caches(isolated_caches, monkeypatch):
    monkeypatch.setattr(config, "WEATHERMAP_API_KEY", "key")
    payload = {"main": {"temp": 71.6, "feels_like": 70.1, "humidity": 55},
               "weather": [{"main": "Clouds", "description": "scattered clouds"}],
               "wind": {"speed": 9.2}, "sys": {"sunrise": 1700000000, "sunset": 1700040000}, "name": "Waupun"}
    monkeypatch.setattr(weather, "_fetch", lambda endpoint, city: payload)
    current = weather.get_current_weather()
    assert current["temp"] == 71.6 and current["description"] == "scattered clouds"
    monkeypatch.setattr(weather, "_fetch", lambda endpoint, city: {"garbage": True})
    assert weather.get_current_weather() == current


def test_current_weather_without_key_is_none(isolated_caches, monkeypatch):
    monkeypatch.setattr(config, "WEATHERMAP_API_KEY", "")
    assert weather.get_current_weather() is None
    assert weather.get_5day_forecast() is None
    assert weather.get_todays_forecast() == []


def test_icons():
    assert weather.icon_kind("light rain") == "rain"
    assert weather.icon_kind("few clouds") == "partly"
    assert weather.icon_kind("thunderstorm with heavy rain") == "storm"
    assert weather.icon_kind("mystery") is None
    for kind in ("clear sky", "overcast clouds", "snow", "mist", "windy", "light rain", "thunderstorm", "few clouds"):
        icon = weather.get_weather_icon(kind, 64)
        assert icon.shape == (64, 64, 4) and icon.dtype == np.uint8 and icon[..., 3].max() == 255
    assert weather.get_weather_icon("mystery") is None
