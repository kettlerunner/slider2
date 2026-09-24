"""Video playback with real-time pacing and first-frame extraction."""

import time

import cv2
import numpy as np

from slider import config
from slider.image_processing import resize_and_pad, resize_to_fit, to_bgr


def get_first_frame(video_path):
    """The first frame of a video, fitted to the display, or None."""
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            print(f"Could not open video {video_path}")
            return None
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        print(f"Could not read first frame of video {video_path}")
        return None
    return resize_and_pad(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)


def play_video(video_path, present_fn, wait_fn, stop_check=None):
    """Play a video, calling present_fn(frame) per frame at the file's frame rate.

    wait_fn(ms) pumps the GUI and returns the pressed key. Frames are dropped
    when decoding falls behind so playback stays in sync with the clock.

    Returns (last_frame, quit_requested).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return get_first_frame(video_path), False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 240:
        fps = 30.0
    frame_interval = 1.0 / fps

    width, height = config.FRAME_WIDTH, config.FRAME_HEIGHT
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    placement = None  # (top, left, fitted_w, fitted_h), computed from the first frame
    last_frame = None
    quit_requested = False
    start = time.monotonic()
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame = to_bgr(frame)
            if placement is None:
                fitted = resize_to_fit(frame, width, height)
                fh, fw = fitted.shape[:2]
                placement = ((height - fh) // 2, (width - fw) // 2, fw, fh)
            top, left, fw, fh = placement
            if frame.shape[1] != fw or frame.shape[0] != fh:
                frame = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA if frame.shape[1] > fw else cv2.INTER_LINEAR)
            canvas[top:top + fh, left:left + fw] = frame
            present_fn(canvas)
            last_frame = canvas

            frame_index += 1
            target = start + frame_index * frame_interval
            delay_ms = int((target - time.monotonic()) * 1000)
            key = wait_fn(max(1, delay_ms))
            if key in (ord("q"), ord("Q"), 27):
                quit_requested = True
                break
            if stop_check and stop_check():
                break

            # Fell behind by more than half a second: skip decoding to catch up.
            while time.monotonic() - target > 0.5:
                if not cap.grab():
                    break
                frame_index += 1
                target = start + frame_index * frame_interval
    finally:
        cap.release()

    if last_frame is None:
        return get_first_frame(video_path), quit_requested
    return last_frame.copy(), quit_requested
