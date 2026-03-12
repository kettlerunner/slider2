"""Camera-based ambient brightness detection for auto-dimming."""

import cv2

from slider import config

_camera = None


def get_ambient_brightness():
    """Capture a frame from the camera and return its average brightness (0-255).

    Returns None on error.
    """
    global _camera

    if _camera is None:
        _camera = cv2.VideoCapture(config.CAMERA_INDEX)
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
