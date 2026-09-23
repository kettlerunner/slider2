"""Background workers: everything slow (network, disk, camera) runs off the render thread.

Three daemon threads publish into a SharedState the main loop reads each frame:

* data   - current weather, forecasts, AI weather summary, AI news digest
* media  - Google Drive sync and pre-scaled display copies of images
* sensor - ambient brightness from the camera
"""

import threading
import time
import traceback
from types import SimpleNamespace

from slider import config, forecast_summary, news, weather
from slider.brightness import AmbientSensor
from slider.drive_sync import (
    authenticate_drive,
    load_local_metadata,
    load_media_from_local_cache,
    refresh_media_items,
)
from slider.image_processing import prepare_display_copy, prune_display_copies

DRIVE_AUTH_RETRY_SECONDS = 600
PUBLISH_PROGRESS_SECONDS = 5


class SharedState:
    """Thread-safe bag of the latest data. Readers take a snapshot each frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._data = {
            "weather": None,            # dict from weather.get_current_weather()
            "forecast_5day": None,      # list from weather.get_5day_forecast()
            "today_forecast": [],       # list from weather.get_todays_forecast()
            "forecast_summary": None,   # (text, style) or None
            "news_pool": [],            # list of story dicts
            "news_fetched_at": None,    # datetime
            "ambient_dark": False,
            "media_items": None,        # None until the first sync/scan completes
            "media_version": 0,
            "drive_connected": False,
        }

    def update(self, **fields):
        with self._lock:
            self._data.update(fields)

    def get(self, name):
        with self._lock:
            return self._data[name]

    def publish_media(self, items):
        with self._lock:
            self._data["media_items"] = list(items)
            self._data["media_version"] += 1

    def snapshot(self):
        with self._lock:
            return SimpleNamespace(**self._data)


class Worker(threading.Thread):
    """Runs a function every `interval` seconds until stopped; errors never kill it."""

    def __init__(self, name, interval_seconds, task, stop_event, run_immediately=True):
        super().__init__(name=name, daemon=True)
        self.interval = max(0.5, float(interval_seconds))
        self.task = task
        self.stop_event = stop_event
        self.run_immediately = run_immediately

    def run(self):
        if not self.run_immediately:
            self.stop_event.wait(self.interval)
        while not self.stop_event.is_set():
            started = time.monotonic()
            try:
                self.task()
            except Exception as exc:
                print(f"[{self.name}] task failed: {exc}")
                traceback.print_exc()
            elapsed = time.monotonic() - started
            self.stop_event.wait(max(0.5, self.interval - elapsed))


# ---------------------------------------------------------------------------
# Data worker
# ---------------------------------------------------------------------------

def _refresh_data(state):
    """Refresh weather, forecasts, AI summary, and news. Each step is independent."""
    if config.DEMO_MODE:
        from slider.demo import apply_demo_data
        apply_demo_data(state)
        return

    try:
        state.update(weather=weather.get_current_weather())
    except Exception as exc:
        print(f"Weather refresh failed: {exc}")

    today, five_day = [], None
    try:
        today = weather.get_todays_forecast()
        five_day = weather.get_5day_forecast()
        state.update(today_forecast=today, forecast_5day=five_day)
    except Exception as exc:
        print(f"Forecast refresh failed: {exc}")

    try:
        context = weather.build_ai_context(state.get("weather"), today, five_day)
        state.update(forecast_summary=forecast_summary.generate(context))
    except Exception as exc:
        print(f"Forecast summary failed: {exc}")

    try:
        news.refresh_news()
        pool, fetched_at = news.get_news_pool()
        state.update(news_pool=pool, news_fetched_at=fetched_at)
    except Exception as exc:
        print(f"News refresh failed: {exc}")


# ---------------------------------------------------------------------------
# Media worker
# ---------------------------------------------------------------------------

class MediaSync:
    def __init__(self, state):
        self.state = state
        self.service = None
        self.next_auth_attempt = 0.0
        self.metadata = load_local_metadata(config.METADATA_FILE)
        self.scaled_initial_cache = False

    def _attach_display_copies(self, items):
        """Fill display_path for image items, publishing progress every few seconds."""
        last_publish = time.monotonic()
        for item in items:
            if item.get("type") != "image":
                continue
            scaled = prepare_display_copy(item["path"])
            if scaled:
                item["display_path"] = scaled
            if time.monotonic() - last_publish > PUBLISH_PROGRESS_SECONDS:
                self.state.publish_media(items)
                last_publish = time.monotonic()
        prune_display_copies([i["path"] for i in items if i.get("type") == "image"])

    def run(self):
        if not self.scaled_initial_cache:
            self.scaled_initial_cache = True
            local_items = load_media_from_local_cache(config.IMAGES_DIR)
            if local_items:
                self.state.publish_media(local_items)
                self._attach_display_copies(local_items)
                self.state.publish_media(local_items)

        if self.service is None:
            now = time.monotonic()
            if now < self.next_auth_attempt:
                return
            self.service = authenticate_drive()
            if self.service is None:
                self.next_auth_attempt = now + DRIVE_AUTH_RETRY_SECONDS
                if self.state.get("media_items") is None:
                    self.state.publish_media(load_media_from_local_cache(config.IMAGES_DIR))
                return
            self.state.update(drive_connected=True)

        items, metadata, downloaded = refresh_media_items(
            self.service, config.DRIVE_FOLDER_ID, config.IMAGES_DIR, config.METADATA_FILE, self.metadata,
        )
        if items is None:
            print("Media refresh failed; keeping existing playlist.")
            if self.state.get("media_items") is None:
                self.state.publish_media(load_media_from_local_cache(config.IMAGES_DIR))
            return
        self.metadata = metadata
        if not items:
            print("Drive folder has no media; keeping existing playlist.")
            if self.state.get("media_items") is None:
                self.state.publish_media([])
            return

        # Reuse display copies that already exist so the playlist is usable at once.
        previous = {i["path"]: i for i in (self.state.get("media_items") or [])}
        for item in items:
            prev = previous.get(item["path"])
            if prev and item["name"] not in downloaded and prev.get("display_path"):
                item["display_path"] = prev["display_path"]
        self.state.publish_media(items)
        self._attach_display_copies(items)
        self.state.publish_media(items)
        if downloaded:
            print(f"Media playlist refreshed: {len(items)} items, {len(downloaded)} new.")


# ---------------------------------------------------------------------------
# Sensor worker
# ---------------------------------------------------------------------------

class BrightnessTask:
    def __init__(self, state):
        self.state = state
        self.sensor = AmbientSensor()

    def run(self):
        value = self.sensor.read()
        if value is not None:
            self.state.update(ambient_dark=value < config.BRIGHTNESS_DARK_THRESHOLD)

    def close(self):
        self.sensor.release()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class BackgroundServices:
    def __init__(self, state):
        self.state = state
        self.stop_event = threading.Event()
        self.media = MediaSync(state)
        self.brightness = BrightnessTask(state) if config.CAMERA_ENABLED else None
        self.workers = [
            Worker("media", config.MEDIA_REFRESH_INTERVAL.total_seconds(), self.media.run, self.stop_event),
            Worker("data", config.DATA_POLL_INTERVAL.total_seconds(), lambda: _refresh_data(state), self.stop_event),
        ]
        if self.brightness:
            self.workers.append(
                Worker("sensor", config.BRIGHTNESS_CHECK_INTERVAL.total_seconds(), self.brightness.run, self.stop_event)
            )

    def start(self):
        for worker in self.workers:
            worker.start()

    def stop(self, timeout=2.0):
        self.stop_event.set()
        for worker in self.workers:
            worker.join(timeout)
        if self.brightness:
            self.brightness.close()
