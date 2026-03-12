"""Image resizing, background blurring, stitching, and frame normalization."""

import math
import random

import cv2
import numpy as np

from slider import config


def normalize_frame_for_display(frame, enforce_size=True):
    """Normalize frame data so every render is consistent for fullscreen playback."""
    if frame is None:
        return None

    normalized = frame

    if not isinstance(normalized, np.ndarray):
        return None

    if normalized.dtype != np.uint8:
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

    if normalized.ndim == 2:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    elif normalized.ndim == 3:
        channels = normalized.shape[2]
        if channels == 1:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        elif channels == 4:
            normalized = cv2.cvtColor(normalized, cv2.COLOR_BGRA2BGR)

    if enforce_size and (
        normalized.shape[0] != config.FRAME_HEIGHT or normalized.shape[1] != config.FRAME_WIDTH
    ):
        normalized = cv2.resize(
            normalized, (config.FRAME_WIDTH, config.FRAME_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )

    return np.ascontiguousarray(normalized)


def resize_and_pad(image, width, height):
    """Resize an image to fit in (width, height) with black padding."""
    if image is None:
        return None

    img = image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None

    scale = min(width / float(w), height / float(h))
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    channels = resized.shape[2] if resized.ndim == 3 else 3
    if channels == 4:
        padded = np.zeros((height, width, 4), dtype=np.uint8)
    else:
        padded = np.zeros((height, width, 3), dtype=np.uint8)

    top_pad = (height - resized.shape[0]) // 2
    left_pad = (width - resized.shape[1]) // 2
    padded[top_pad:top_pad + resized.shape[0], left_pad:left_pad + resized.shape[1]] = resized
    return padded


def create_zoomed_blurred_background(image, width, height):
    """Create a zoomed and blurred version of an image as a background."""
    img = image.copy()
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    scale = max(width / float(w), height / float(h)) * 1.1
    new_w, new_h = int(w * scale), int(h * scale)
    zoomed = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    start_x = max((zoomed.shape[1] - width) // 2, 0)
    start_y = max((zoomed.shape[0] - height) // 2, 0)
    cropped = zoomed[start_y:start_y + height, start_x:start_x + width]

    blurred = cv2.GaussianBlur(cropped, (31, 31), 0)
    return blurred


def ensure_same_channels(img1, img2):
    """Ensure both images have the same number of channels."""
    if img1 is None or img2 is None:
        return img1, img2

    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    if img2.ndim == 2:
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    if img1.ndim == 3 and img2.ndim == 3 and img1.shape[2] != img2.shape[2]:
        if img1.shape[2] == 3 and img2.shape[2] == 4:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
        elif img1.shape[2] == 4 and img2.shape[2] == 3:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

    return img1, img2


def _to_bgr(image):
    """Convert an image to 3-channel BGR."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def stitch_images(images, width, height):
    """Build a multi-photo collage on top of a zoomed, blurred background."""
    if not images:
        return np.zeros((height, width, 3), dtype=np.uint8)

    bg_src = random.choice(images)
    if bg_src is None or bg_src.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    bg_base = _to_bgr(bg_src)
    background = create_zoomed_blurred_background(bg_base, width, height)

    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(float(n) / cols))

    cell_w = width // cols
    cell_h = height // rows
    margin_factor = 0.9

    for idx, img in enumerate(images):
        if img is None or img.size == 0:
            continue

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.ndim == 3 and img.shape[2] > 4:
            img = img[:, :, :4]

        h, w = img.shape[:2]
        if h == 0 or w == 0:
            continue

        scale = min((cell_w * margin_factor) / float(w), (cell_h * margin_factor) / float(h))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        row = idx // cols
        col = idx % cols
        cell_x = col * cell_w
        cell_y = row * cell_h
        left = cell_x + (cell_w - new_w) // 2
        top = cell_y + (cell_h - new_h) // 2

        if left < 0 or top < 0 or left + new_w > width or top + new_h > height:
            left = max(0, left)
            top = max(0, top)
            new_w = min(new_w, width - left)
            new_h = min(new_h, height - top)
            resized = resized[:new_h, :new_w]

        roi = background[top:top + new_h, left:left + new_w]

        if resized.ndim == 3 and resized.shape[2] == 4:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR).astype(np.float32)
            alpha = resized[:, :, 3].astype(np.float32) / 255.0
            alpha = alpha[:, :, np.newaxis]
            roi_f = roi.astype(np.float32)
            blended = roi_f * (1.0 - alpha) + rgb * alpha
            background[top:top + new_h, left:left + new_w] = blended.astype(np.uint8)
        else:
            background[top:top + new_h, left:left + new_w] = resized

    return background.astype(np.uint8)


def create_single_image_with_background(image, width, height):
    """Place an image centered on a zoomed, blurred background of itself."""
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    bg_base = _to_bgr(image)
    background = create_zoomed_blurred_background(bg_base, width, height)

    img = image
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] > 4:
        img = img[:, :, :4]

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return background

    margin_factor = 0.9
    scale = min((width * margin_factor) / float(w), (height * margin_factor) / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (height - new_h) // 2
    left = (width - new_w) // 2

    if resized.ndim == 3 and resized.shape[2] == 4:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
        alpha = resized[:, :, 3].astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]
        roi = background[top:top + new_h, left:left + new_w].astype(np.float32)
        blended = roi * (1.0 - alpha) + rgb.astype(np.float32) * alpha
        background[top:top + new_h, left:left + new_w] = blended.astype(np.uint8)
    else:
        background[top:top + new_h, left:left + new_w] = resized

    return background.astype(np.uint8)
