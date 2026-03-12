"""Main slideshow loop — media cycling, transitions, display frame building, and supervisor."""

import random
import time
import traceback
from datetime import datetime

import cv2

from slider import config
from slider.utils import get_central_time
from slider.image_processing import create_zoomed_blurred_background, create_single_image_with_background, stitch_images
from slider.overlays import add_forecast_overlay, add_news_overlay, add_quote_overlay
from slider.display import ensure_fullscreen, show_frame, present_frame
from slider.cursor import hide_mouse_cursor, show_mouse_cursor
from slider.brightness import get_ambient_brightness, release_camera
from slider.touch_ui import build_mode_buttons
from slider.transitions import BASIC_TRANSITIONS, ALL_TRANSITIONS, fade_transition
from slider.drive_sync import authenticate_drive, refresh_media_items, load_local_metadata, load_media_from_local_cache
from slider.weather import get_current_weather, get_todays_forecast, get_5day_forecast
from slider.news import get_ai_generated_news
from slider.forecast_summary import get_or_generate_forecast_summary, STYLE_TITLES
from slider.quotes import get_random_quote
from slider.video import get_first_frame, play_video

# ---------------------------------------------------------------------------
# Image loading cache
# ---------------------------------------------------------------------------

_image_cache = {}


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

    max_dim = max(config.FRAME_WIDTH, config.FRAME_HEIGHT) * 2
    scale = min(max_dim / float(w), max_dim / float(h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w > 0 and new_h > 0:
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if len(_image_cache) >= config.MAX_IMAGE_CACHE:
        try:
            _image_cache.pop(next(iter(_image_cache)))
        except StopIteration:
            pass

    _image_cache[path] = img
    return img


# ---------------------------------------------------------------------------
# Display frame builders
# ---------------------------------------------------------------------------

def build_display_frame(media_item, image_paths, forecast_5day):
    """Build a display frame for an image media item.

    Chooses between single, stitch, quote, forecast, today, news
    depending on time of day and day of week.
    """
    if media_item["type"] != "image":
        return None

    base_img = load_scaled_image(media_item["path"])
    if base_img is None:
        return None

    ct = get_central_time()
    current_hour = ct.hour
    current_day = ct.weekday()

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

    fw, fh = config.FRAME_WIDTH, config.FRAME_HEIGHT

    if display_type == "forecast":
        frame = create_zoomed_blurred_background(base_img, fw, fh)
        frame = add_forecast_overlay(frame, forecast_5day)
        return frame

    if display_type == "stitch":
        stitch_count = random.randint(2, 4)
        other_paths = [p for p in image_paths if p != media_item["path"]]
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
            return create_single_image_with_background(base_img, fw, fh)
        return stitch_images(images, fw, fh)

    if display_type == "quote":
        frame = create_zoomed_blurred_background(base_img, fw, fh)
        quote, source = get_random_quote()
        frame = add_quote_overlay(frame, quote, source)
        return frame

    if display_type == "today":
        frame = create_zoomed_blurred_background(base_img, fw, fh)
        weather_data = get_todays_forecast()

        forecast_summary = "Weather summary unavailable."
        custom_title = "Today's Forecast"

        if weather_data:
            forecast_summary, style_used = get_or_generate_forecast_summary(weather_data, style="random")
            custom_title = STYLE_TITLES.get(style_used, "Today's Forecast")

        frame = add_quote_overlay(
            frame, quote=forecast_summary, source="Today's Weather",
            title=custom_title, style=None,
        )
        return frame

    if display_type == "news":
        frame = create_zoomed_blurred_background(base_img, fw, fh)
        news = get_ai_generated_news()
        frame = add_news_overlay(frame, news)
        return frame

    # Default: single image
    return create_single_image_with_background(base_img, fw, fh)


def build_display_frame_for_mode(media_item, image_paths, forecast_5day, mode):
    """Build a display frame based on the active touch-UI mode."""
    if media_item["type"] != "image":
        return None

    if mode == "random":
        return build_display_frame(media_item, image_paths, forecast_5day)

    base_img = load_scaled_image(media_item["path"])
    if base_img is None:
        return None

    fw, fh = config.FRAME_WIDTH, config.FRAME_HEIGHT

    if mode == "news":
        frame = create_zoomed_blurred_background(base_img, fw, fh)
        news = get_ai_generated_news()
        return add_news_overlay(frame, news)

    if mode == "weather":
        frame = create_zoomed_blurred_background(base_img, fw, fh)
        if forecast_5day and random.random() < 0.5:
            return add_forecast_overlay(frame, forecast_5day)

        weather_data = get_todays_forecast()
        forecast_summary = "Weather summary unavailable."
        custom_title = "Today's Forecast"

        if weather_data:
            forecast_summary, style_used = get_or_generate_forecast_summary(
                weather_data, style="random"
            )
            custom_title = STYLE_TITLES.get(style_used, "Today's Forecast")

        return add_quote_overlay(
            frame, quote=forecast_summary, source="Today's Weather",
            title=custom_title, style=None,
        )

    if mode == "pictures":
        return create_single_image_with_background(base_img, fw, fh)

    return build_display_frame(media_item, image_paths, forecast_5day)


# ---------------------------------------------------------------------------
# Main slideshow session
# ---------------------------------------------------------------------------

def run_slideshow_once():
    """Run one full slideshow session.

    Returns True if the user requested exit (pressing 'q'),
    otherwise False (to allow restart on error).
    """
    folder_id = config.DRIVE_FOLDER_ID
    temp_dir = config.resource_path("images")
    metadata_file = config.resource_path("metadata.json")

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

    forecast_5day = get_5day_forecast() or []

    if config.LOW_POWER_MODE:
        transitions = [fade_transition]
        print("Low-power mode enabled: using simplified transitions.")
    else:
        transitions = ALL_TRANSITIONS

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

    current_frame = None

    def get_next_index(current_idx):
        nonlocal play_queue, active_items
        total = len(active_items)
        if total <= 1:
            return current_idx
        play_queue = [i for i in play_queue if 0 <= i < total and i != current_idx]
        if not play_queue:
            play_queue = [i for i in range(total) if i != current_idx]
            random.shuffle(play_queue)
        return play_queue.pop(0)

    def get_mode_label(mode):
        for entry in config.MODE_DEFINITIONS:
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

            # Ambient brightness check
            if now - last_brightness_check >= config.BRIGHTNESS_CHECK_INTERVAL:
                b = get_ambient_brightness()
                last_brightness_check = now
                if b is not None:
                    ambient_dark = b < config.BRIGHTNESS_DARK_THRESHOLD

            # Periodic Drive refresh
            if service is not None and now - last_refresh_time >= config.MEDIA_REFRESH_INTERVAL:
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

            # Occasionally refresh 5-day forecast
            if now.minute == 0 and now.second < 5:
                refreshed = get_5day_forecast()
                if refreshed is not None:
                    forecast_5day = refreshed

            temp, weather = get_current_weather()
            status_text = f"Mode: {get_mode_label(mode_state['mode'])}"
            if mode_state["fallback"]:
                status_text += " (fallback: no items)"

            # --- Display current item ---
            try:
                if current_item["type"] == "video":
                    def _present_video_frame(frame):
                        present_frame(
                            frame, temp, weather,
                            status_text=status_text,
                            ambient_dark=ambient_dark,
                            ui_state=mode_state,
                        )

                    current_frame, quit_requested = play_video(
                        current_item["path"],
                        _present_video_frame,
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
                            print(f"Video {current_item['name']} could not be decoded; skipping.")
                else:
                    if current_frame is None:
                        frame = build_display_frame_for_mode(
                            current_item, active_image_paths, forecast_5day, mode_state["mode"]
                        )
                        if frame is None:
                            print(f"Image {current_item['name']} could not be loaded; skipping.")
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
                        current_frame, temp, weather,
                        status_text=status_text,
                        ambient_dark=ambient_dark,
                        ui_state=mode_state,
                    )
                    wait_start = time.time()
                    while time.time() - wait_start < config.DISPLAY_TIME:
                        if mode_state["show_buttons"] or mode_state["needs_redraw"]:
                            present_frame(
                                current_frame, temp, weather,
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
                print(f"Error displaying item {current_item.get('name', '<unknown>')}: {exc}")
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
                    print(f"Failed to prepare next frame for {next_item['name']}. Skipping transition.")
                else:
                    if current_frame is None:
                        current_frame = next_frame

                    transition_fn = random.choice(transitions)
                    for frame in transition_fn(current_frame, next_frame, config.NUM_TRANSITION_FRAMES):
                        present_frame(
                            frame, temp, weather,
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

                    current_frame = next_frame

                current_item = next_item
                index = next_index
            else:
                continue

    finally:
        release_camera()
        show_mouse_cursor()
        cv2.destroyAllWindows()

    return exit_requested


# ---------------------------------------------------------------------------
# Supervisor loop
# ---------------------------------------------------------------------------

def main():
    """Run the slideshow with automatic restart on errors."""
    consecutive_failures = 0
    base_delay = 5

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
