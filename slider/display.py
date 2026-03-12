"""Window management, fullscreen enforcement, and frame presentation."""

import time

import cv2
import numpy as np

from slider import config
from slider.image_processing import normalize_frame_for_display
from slider.overlays import add_time_overlay
from slider.touch_ui import draw_mode_buttons


# ---------------------------------------------------------------------------
# Fullscreen management
# ---------------------------------------------------------------------------

_window_fullscreen_state = {}


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
            cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)
            window_state["is_fullscreen"] = True
        elif not window_state.get("is_fullscreen", False):
            window_state["is_fullscreen"] = True
    except cv2.error as exc:
        print(f"Failed to enforce fullscreen for '{window_name}': {exc}")


# ---------------------------------------------------------------------------
# Frame display
# ---------------------------------------------------------------------------

def show_frame(window_name: str, frame):
    """Display a frame and re-assert fullscreen."""
    prepared_frame = normalize_frame_for_display(frame)
    if prepared_frame is None:
        return
    cv2.imshow(window_name, prepared_frame)
    ensure_fullscreen(window_name)


# ---------------------------------------------------------------------------
# Button visibility
# ---------------------------------------------------------------------------

def _update_button_visibility(ui_state):
    """Auto-hide mode buttons after timeout."""
    if not ui_state:
        return
    if not ui_state.get("show_buttons"):
        return
    last_touch = ui_state.get("last_touch")
    if last_touch is None:
        return
    if time.monotonic() - last_touch >= config.BUTTON_AUTOHIDE_SECONDS:
        ui_state["show_buttons"] = False


# ---------------------------------------------------------------------------
# Present frame (final output step)
# ---------------------------------------------------------------------------

def present_frame(frame, temp, weather, status_text=None, ambient_dark=False, ui_state=None):
    """Final step before putting pixels on the screen.

    If ambient_dark is True, shows a black screen (room lights are off).
    Otherwise, overlays time/weather and shows the frame.
    """
    _update_button_visibility(ui_state)
    if ambient_dark:
        black = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        show_frame("slideshow", black)
    else:
        overlay = add_time_overlay(frame, temp, weather, status_text=status_text)
        if ui_state and ui_state.get("show_buttons") and overlay is not None:
            overlay = draw_mode_buttons(overlay, ui_state["buttons"], ui_state["mode"])
        show_frame("slideshow", overlay)
