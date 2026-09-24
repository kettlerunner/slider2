"""Window management and frame presentation (the only module that calls imshow)."""

import os
import time

import cv2
import numpy as np

from slider import config
from slider.image_processing import normalize_frame_for_display
from slider.overlays import draw_hud
from slider.touch_ui import hit_test
from slider.utils import now_local

WINDOW_NAME = "slideshow"
QUIT_KEYS = {ord("q"), ord("Q"), 27}  # q or Esc


class HudState:
    """Everything the status bar and touch buttons need, owned by the main loop."""

    def __init__(self, buttons):
        self.buttons = buttons
        self.mode = "random"
        self.mode_dirty = False
        self.show_buttons = False
        self.last_touch = None
        self.needs_redraw = False
        self.weather = None
        self.status_text = ""
        self.ambient_dark = False

    def touch(self, x, y):
        """Handle a tap. The first tap reveals the buttons; later taps select."""
        now = time.monotonic()
        self.last_touch = now
        self.needs_redraw = True
        if not self.show_buttons:
            self.show_buttons = True
            return
        new_mode = hit_test(self.buttons, x, y)
        if new_mode and new_mode != self.mode:
            self.mode = new_mode
            self.mode_dirty = True

    def tick(self):
        """Auto-hide the buttons after the configured idle time."""
        if self.show_buttons and self.last_touch is not None:
            if time.monotonic() - self.last_touch >= config.BUTTON_AUTOHIDE_SECONDS:
                self.show_buttons = False
                self.needs_redraw = True


class Window:
    """The OpenCV window. Must be used from the main thread only."""

    def __init__(self, name=WINDOW_NAME):
        self.name = name
        self._last_fullscreen_check = 0.0
        self._black = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        if config.WINDOWED:
            cv2.resizeWindow(name, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        else:
            self.ensure_fullscreen(force=True)

    def ensure_fullscreen(self, force=False):
        """Re-assert fullscreen, at most every couple of seconds (window managers can undo it)."""
        if config.WINDOWED:
            return
        now = time.monotonic()
        if not force and now - self._last_fullscreen_check < 2.0:
            return
        self._last_fullscreen_check = now
        try:
            flag = getattr(cv2, "WINDOW_FULLSCREEN", 1)
            try:
                is_fullscreen = int(cv2.getWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN)) == flag
            except cv2.error:
                is_fullscreen = False
            if not is_fullscreen:
                cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, flag)
                cv2.moveWindow(self.name, 0, 0)
                cv2.resizeWindow(self.name, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        except cv2.error as exc:
            print(f"Failed to enforce fullscreen: {exc}")

    def show(self, frame):
        if frame is None:
            return
        if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
                or frame.shape[:2] != (config.FRAME_HEIGHT, config.FRAME_WIDTH)):
            frame = normalize_frame_for_display(frame)
            if frame is None:
                return
        cv2.imshow(self.name, frame)
        self._capture(frame)
        self.ensure_fullscreen()

    def show_black(self):
        cv2.imshow(self.name, self._black)
        self._capture(self._black)
        self.ensure_fullscreen()

    _capture_count = 0
    _last_capture = 0.0

    def _capture(self, frame):
        """Development aid: periodically save what is on screen (SLIDER_CAPTURE_DIR)."""
        if not config.CAPTURE_DIR:
            return
        now = time.monotonic()
        if now - self._last_capture < config.CAPTURE_SECONDS:
            return
        self._last_capture = now
        try:
            os.makedirs(config.CAPTURE_DIR, exist_ok=True)
            self._capture_count += 1
            cv2.imwrite(os.path.join(config.CAPTURE_DIR, f"frame_{self._capture_count:04d}.png"), frame)
        except (OSError, cv2.error) as exc:
            print(f"Capture failed: {exc}")

    def poll(self, delay_ms=1):
        """Pump the GUI event loop and return the pressed key (or -1)."""
        key = cv2.waitKey(max(1, int(delay_ms)))
        return key & 0xFF if key >= 0 else -1

    def set_mouse_callback(self, callback):
        cv2.setMouseCallback(self.name, callback)

    def close(self):
        try:
            cv2.destroyWindow(self.name)
        except cv2.error:
            pass
        cv2.destroyAllWindows()


def present(window, frame, hud, now=None):
    """Put a slide on screen with the HUD (or a black screen when the room is dark)."""
    hud.tick()
    if hud.ambient_dark:
        window.show_black()
    else:
        window.show(draw_hud(frame, hud, now or now_local()))
