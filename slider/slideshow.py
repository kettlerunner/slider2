"""Main slideshow loop: picks slides, renders them, runs transitions, handles touch.

Everything slow happens in slider.services threads; this loop only reads the
SharedState snapshot, builds frames from pre-scaled images, and draws.
"""

import random
import time
import traceback

import cv2
import numpy as np

from slider import config
from slider.cursor import hide_mouse_cursor, show_mouse_cursor
from slider.display import QUIT_KEYS, HudState, Window, present
from slider.forecast_summary import STYLE_TITLES, UNAVAILABLE
from slider.image_processing import (
    ImageCache,
    create_single_image_with_background,
    create_zoomed_blurred_background,
    stitch_images,
)
from slider.news import FALLBACK_STORY, pick_story
from slider.overlays import add_forecast_overlay, add_news_overlay, add_quote_overlay
from slider.playlist import Playlist
from slider.quotes import get_random_quote
from slider.services import BackgroundServices, SharedState
from slider.touch_ui import build_mode_buttons, mode_label
from slider.transitions import ALL_TRANSITIONS, LOW_POWER_TRANSITIONS
from slider.utils import now_local
from slider.video import get_first_frame, play_video

# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------

def choose_display_type(snap, now, weights=None):
    """Pick a layout for an image slide, skipping layouts whose data is missing."""
    if weights is None:
        mix = config.DISPLAY_MIX
        if mix["day_start_hour"] <= now.hour < mix["day_end_hour"]:
            weights = mix["weekday"] if now.weekday() < 5 else mix["weekend"]
        else:
            weights = mix["evening"]

    available = {
        "forecast": bool(snap.forecast_5day),
        "today": snap.forecast_summary is not None,
        "news": bool(snap.news_pool),
    }
    choices = [(name, w) for name, w in weights.items() if w > 0 and available.get(name, True)]
    if not choices:
        return "single"
    names, values = zip(*choices)
    return random.choices(names, weights=values, k=1)[0]


class SlideBuilder:
    """Builds display frames for image items from the shared data snapshot."""

    def __init__(self):
        self.cache = ImageCache()
        self.last_headline = None
        self.size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)

    def _load(self, item):
        img = self.cache.get(item.get("display_path") or item["path"])
        if img is None and item.get("display_path") and item["display_path"] != item["path"]:
            img = self.cache.get(item["path"])
        return img

    def news_frame(self, base_img, snap):
        story = pick_story(snap.news_pool, avoid_headline=self.last_headline) or FALLBACK_STORY
        self.last_headline = story.get("headline")
        frame = create_zoomed_blurred_background(base_img, *self.size)
        return add_news_overlay(frame, story, snap.news_fetched_at)

    def today_frame(self, base_img, snap):
        frame = create_zoomed_blurred_background(base_img, *self.size)
        summary = snap.forecast_summary
        if summary:
            text, style = summary
            title = STYLE_TITLES.get(style, "Today's Forecast")
        else:
            text, title = UNAVAILABLE, "Today's Forecast"
        return add_quote_overlay(frame, quote=text, source="", title=title)

    def forecast_frame(self, base_img, snap):
        frame = create_zoomed_blurred_background(base_img, *self.size)
        return add_forecast_overlay(frame, snap.forecast_5day, snap.weather)

    def quote_frame(self, base_img):
        frame = create_zoomed_blurred_background(base_img, *self.size)
        quote, source = get_random_quote()
        return add_quote_overlay(frame, quote, source)

    def stitch_frame(self, base_img, item, image_paths):
        wanted = random.randint(2, 4)
        others = [p for p in image_paths if p != (item.get("display_path") or item["path"])]
        pool = random.sample(others, wanted) if len(others) >= wanted else others
        images = [base_img] + [img for img in (self.cache.get(p) for p in pool) if img is not None]
        random.shuffle(images)
        if len(images) < 2:
            return create_single_image_with_background(base_img, *self.size)
        return stitch_images(images[:wanted], *self.size)

    def build(self, item, image_paths, snap, mode):
        """Return a frame for an image item, or None if it cannot be loaded."""
        if item.get("type") != "image":
            return None
        base_img = self._load(item)
        if base_img is None:
            return None

        if mode == "news":
            return self.news_frame(base_img, snap)
        if mode == "weather":
            if snap.forecast_5day and (snap.forecast_summary is None or random.random() < 0.5):
                return self.forecast_frame(base_img, snap)
            return self.today_frame(base_img, snap)
        if mode == "pictures":
            return create_single_image_with_background(base_img, *self.size)

        display_type = choose_display_type(snap, now_local())
        if display_type == "forecast":
            return self.forecast_frame(base_img, snap)
        if display_type == "stitch":
            return self.stitch_frame(base_img, item, image_paths)
        if display_type == "quote":
            return self.quote_frame(base_img)
        if display_type == "today":
            return self.today_frame(base_img, snap)
        if display_type == "news":
            return self.news_frame(base_img, snap)
        return create_single_image_with_background(base_img, *self.size)


def message_frame(text, title="Slider"):
    frame = np.full((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), 30, dtype=np.uint8)
    return add_quote_overlay(frame, text, source="", title=title)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

def run_slideshow_once():
    """Run one slideshow session. Returns True if the user asked to quit."""
    state = SharedState()
    services = BackgroundServices(state)
    services.start()

    window = Window()
    hide_mouse_cursor()
    hud = HudState(build_mode_buttons())
    window.set_mouse_callback(lambda event, x, y, flags, params: hud.touch(x, y) if event == cv2.EVENT_LBUTTONDOWN else None)

    transitions = LOW_POWER_TRANSITIONS if config.LOW_POWER_MODE else ALL_TRANSITIONS
    if config.LOW_POWER_MODE:
        print("Low-power mode enabled: using lightweight transitions.")

    builder = SlideBuilder()
    playlist = Playlist()
    bad_paths = set()
    seen_media_version = 0
    frame_budget_ms = 1000.0 / config.FPS

    def sync_state():
        """Pull the latest snapshot into the HUD/playlist. Returns the snapshot."""
        nonlocal seen_media_version
        snap = state.snapshot()
        hud.weather = snap.weather
        hud.ambient_dark = snap.ambient_dark
        if snap.media_version != seen_media_version and snap.media_items is not None:
            seen_media_version = snap.media_version
            playlist.set_items([i for i in snap.media_items if i["path"] not in bad_paths])
        return snap

    def status_text():
        text = f"Mode: {mode_label(hud.mode)}"
        if playlist.fallback:
            text += " (fallback: no items)"
        return text

    def wait_for_media():
        while len(playlist) == 0:
            snap = sync_state()
            if len(playlist):
                break
            if snap.media_items is None:
                text = "Waiting for media from Google Drive..."
            elif snap.drive_connected:
                text = "The Drive folder has no photos or videos yet."
            else:
                text = "No local media and Google Drive is unavailable.\nRetrying in the background."
            hud.status_text = ""
            present(window, message_frame(text), hud)
            if window.poll(250) in QUIT_KEYS:
                return False
        return True

    def run_transition(render, duration):
        """Drive a transition from the clock. Returns 'quit', 'dirty', or None."""
        started = time.monotonic()
        while True:
            frame_started = time.monotonic()
            alpha = 1.0 if duration <= 0 else min(1.0, (frame_started - started) / duration)
            present(window, render(alpha), hud)
            spent_ms = (time.monotonic() - frame_started) * 1000.0
            key = window.poll(max(1, int(frame_budget_ms - spent_ms)))
            if key in QUIT_KEYS:
                return "quit"
            if hud.mode_dirty:
                return "dirty"
            if alpha >= 1.0:
                return None

    def hold(frame, seconds):
        """Show a still frame for `seconds`, redrawing only when something changed."""
        deadline = time.monotonic() + seconds
        last_minute = now_local().minute
        last_weather = hud.weather
        last_dark = hud.ambient_dark
        while time.monotonic() < deadline:
            key = window.poll(100)
            if key in QUIT_KEYS:
                return "quit"
            if hud.mode_dirty:
                return "dirty"
            sync_state()
            now = now_local()
            changed = (
                hud.show_buttons or hud.needs_redraw or now.minute != last_minute
                or hud.weather is not last_weather or hud.ambient_dark != last_dark
            )
            if changed:
                hud.status_text = status_text()
                present(window, frame, hud, now)
                hud.needs_redraw = False
                last_minute, last_weather, last_dark = now.minute, hud.weather, hud.ambient_dark
        return None

    exit_requested = False
    try:
        if not wait_for_media():
            return True
        print("Slideshow ready.")

        current_frame = None
        consecutive_errors = 0

        while True:
            snap = sync_state()
            if hud.mode_dirty:
                hud.mode_dirty = False
                playlist.set_mode(hud.mode)
                current_frame = None
                print(f"Mode switched to {mode_label(hud.mode)}.")
            hud.status_text = status_text()

            item = playlist.current
            if item is None:
                if not wait_for_media():
                    exit_requested = True
                    break
                continue

            try:
                # ---- show the current item -------------------------------------
                if item["type"] == "video":
                    if hud.ambient_dark:
                        present(window, None, hud)
                        outcome = hold(None, min(config.DISPLAY_TIME, 5))
                    else:
                        last_frame, quit_requested = play_video(
                            item["path"],
                            lambda frame: present(window, frame, hud),
                            window.poll,
                            stop_check=lambda: hud.mode_dirty,
                        )
                        outcome = "quit" if quit_requested else ("dirty" if hud.mode_dirty else None)
                        current_frame = last_frame
                        if last_frame is None and outcome is None:
                            print(f"Video {item['name']} could not be decoded; removing it.")
                            bad_paths.add(item["path"])
                            playlist.remove(item)
                            continue
                else:
                    if current_frame is None:
                        current_frame = builder.build(item, playlist.image_paths, snap, hud.mode)
                        if current_frame is None:
                            print(f"Image {item['name']} could not be loaded; removing it.")
                            bad_paths.add(item["path"])
                            playlist.remove(item)
                            continue
                    present(window, current_frame, hud)

                    # Build the next slide now so the transition starts on time.
                    next_item = playlist.peek_next()
                    next_frame = None
                    if next_item is not None and not hud.ambient_dark:
                        next_frame = _build_any(builder, next_item, playlist, snap, hud.mode)

                    outcome = hold(current_frame, config.DISPLAY_TIME)
                    if outcome is None and playlist.peek_next() is not next_item:
                        next_frame = None  # playlist changed while we waited

                if outcome == "quit":
                    exit_requested = True
                    break
                if outcome == "dirty":
                    continue

                # ---- transition to the next item ---------------------------------
                next_item = playlist.peek_next()
                if next_item is None:
                    current_frame = None  # single item: pick a fresh layout next time
                    continue

                if item["type"] == "video" or next_frame is None:
                    next_frame = _build_any(builder, next_item, playlist, sync_state(), hud.mode)
                if next_frame is None:
                    print(f"Could not prepare {next_item['name']}; removing it.")
                    bad_paths.add(next_item["path"])
                    playlist.remove(next_item)
                    continue

                if hud.ambient_dark or current_frame is None:
                    present(window, next_frame, hud)
                else:
                    render = random.choice(transitions)(current_frame, next_frame)
                    outcome = run_transition(render, config.TRANSITION_TIME)
                    if outcome == "quit":
                        exit_requested = True
                        break
                    if outcome == "dirty":
                        continue

                current_frame = next_frame
                playlist.advance()
                consecutive_errors = 0

            except KeyboardInterrupt:
                exit_requested = True
                break
            except Exception as exc:
                consecutive_errors += 1
                print(f"Error displaying {item.get('name', '<unknown>')}: {exc}")
                traceback.print_exc()
                if consecutive_errors >= 3:
                    return False  # let the supervisor restart the session
                current_frame = None
                playlist.advance()
    finally:
        services.stop()
        show_mouse_cursor()
        window.close()

    return exit_requested


def _build_any(builder, item, playlist, snap, mode):
    if item["type"] == "video":
        return get_first_frame(item["path"])
    return builder.build(item, playlist.image_paths, snap, mode)


# ---------------------------------------------------------------------------
# Supervisor loop
# ---------------------------------------------------------------------------

def _quiet_opencv_logs():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except Exception:
        pass


def main():
    """Run the slideshow forever, restarting with backoff after fatal errors."""
    _quiet_opencv_logs()
    consecutive_failures = 0
    base_delay = 5

    while True:
        try:
            if run_slideshow_once():
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

        wait_seconds = base_delay if consecutive_failures == 0 else min(60, base_delay * (consecutive_failures + 1))
        print(f"Restarting slideshow in {wait_seconds} seconds...")
        time.sleep(wait_seconds)
