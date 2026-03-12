"""Video playback and first-frame extraction."""

import cv2

from slider import config
from slider.image_processing import resize_and_pad


def get_first_frame(video_path):
    """Extract the first frame from a video, resized to display dimensions."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"Could not read first frame of video {video_path}")
        return None
    frame = resize_and_pad(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)
    return frame


def play_video(video_path, present_fn, stop_check=None):
    """Play a video file, calling present_fn(frame) for each frame.

    Args:
        video_path: Path to the video file.
        present_fn: Callable(frame) to display each frame.
        stop_check: Optional callable returning True to stop early.

    Returns:
        (last_frame, quit_requested) tuple.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video {video_path}")
        return get_first_frame(video_path), False

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 1:
        fps = 30
    wait = int(1000 / fps)

    last_frame = None
    quit_requested = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_and_pad(frame, config.FRAME_WIDTH, config.FRAME_HEIGHT)
        if frame is None:
            continue
        last_frame = frame.copy()
        present_fn(frame)
        key = cv2.waitKey(wait)
        if key == ord("q"):
            quit_requested = True
            break
        if stop_check and stop_check():
            break

    cap.release()

    if last_frame is None:
        last_frame = get_first_frame(video_path)

    return last_frame, quit_requested
