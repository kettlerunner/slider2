"""Camera-based ambient brightness detection for auto-dimming."""

import time

import cv2

from slider import config


class AmbientSensor:
    """Reads average brightness (0-255) from a camera, with backoff when it fails."""

    def __init__(self, index=None):
        self.index = config.CAMERA_INDEX if index is None else index
        self._cap = None
        self._next_open_attempt = 0.0
        self._failures = 0

    def _open(self):
        now = time.monotonic()
        if now < self._next_open_attempt:
            return False
        cap = cv2.VideoCapture(self.index)
        if not cap.isOpened():
            cap.release()
            self._next_open_attempt = now + config.CAMERA_RETRY_INTERVAL.total_seconds()
            print(f"Unable to open camera {self.index} for brightness detection; "
                  f"retrying in {int(config.CAMERA_RETRY_INTERVAL.total_seconds())}s.")
            return False
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except cv2.error:
            pass
        self._cap = cap
        self._failures = 0
        return True

    def read(self):
        """Return the mean brightness of a fresh frame, or None if unavailable."""
        if self._cap is None and not self._open():
            return None
        # Drivers buffer a few frames; grab past the stale ones first.
        self._cap.grab()
        ok, frame = self._cap.read()
        if not ok or frame is None:
            self._failures += 1
            if self._failures >= 3:
                print("Camera stopped delivering frames; will reopen later.")
                self.release()
                self._next_open_attempt = time.monotonic() + config.CAMERA_RETRY_INTERVAL.total_seconds()
            return None
        self._failures = 0
        try:
            gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(gray.mean())
        except cv2.error as exc:
            print(f"Error computing ambient brightness: {exc}")
            return None

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


_sensor = None


def get_ambient_brightness():
    """Module-level convenience wrapper around a shared AmbientSensor."""
    global _sensor
    if _sensor is None:
        _sensor = AmbientSensor()
    return _sensor.read()


def release_camera():
    global _sensor
    if _sensor is not None:
        _sensor.release()
        _sensor = None
