"""Image loading, scaling, backgrounds, collages, and frame normalization.

Every builder returns a BGR uint8 frame of exactly (height, width) so the
transitions and overlays never need to convert or resize.
"""

import math
import os
import random
from collections import OrderedDict

import cv2
import numpy as np

from slider import config

_ALPHA_SOURCE_EXTENSIONS = (".png", ".webp", ".gif")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_first_video_frame(path):
    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return None
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def read_image(path):
    """Read an image as BGR or BGRA uint8, honoring EXIF orientation. None on failure."""
    ext = os.path.splitext(path)[1].lower()
    # IMREAD_UNCHANGED keeps alpha but ignores EXIF orientation, so only use it
    # for formats that can carry transparency.
    flags = cv2.IMREAD_UNCHANGED if ext in _ALPHA_SOURCE_EXTENSIONS else cv2.IMREAD_COLOR
    img = cv2.imread(path, flags)
    if img is None and ext in (".gif", ".webp"):
        img = _read_first_video_frame(path)
    if img is None or img.size == 0:
        return None

    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] > 4:
        img = np.ascontiguousarray(img[:, :, :4])
    return img


def downscale_to(image, max_dim):
    """Shrink so the longest edge is at most max_dim (never upscales)."""
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = max_dim / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def load_image(path, max_dim=None):
    """Read an image and cap its longest edge (default: config.IMAGE_MAX_DIM)."""
    img = read_image(path)
    if img is None:
        print(f"Failed to read image: {path}")
        return None
    return downscale_to(img, max_dim or config.IMAGE_MAX_DIM)


class ImageCache:
    """Small LRU cache of decoded images keyed by path."""

    def __init__(self, capacity=None):
        self.capacity = capacity or config.MAX_IMAGE_CACHE
        self._items = OrderedDict()

    def get(self, path):
        img = self._items.get(path)
        if img is not None:
            self._items.move_to_end(path)
            return img
        img = load_image(path)
        if img is None:
            return None
        self._items[path] = img
        while len(self._items) > self.capacity:
            self._items.popitem(last=False)
        return img

    def invalidate(self, path):
        self._items.pop(path, None)

    def clear(self):
        self._items.clear()


# ---------------------------------------------------------------------------
# Pre-scaled display copies (built once, in the background)
# ---------------------------------------------------------------------------

def scaled_path_for(src_path, dst_dir=None):
    """Where the display-sized copy of src_path lives."""
    dst_dir = dst_dir or config.SCALED_DIR
    name, ext = os.path.splitext(os.path.basename(src_path))
    out_ext = ".png" if ext.lower() in _ALPHA_SOURCE_EXTENSIONS else ".jpg"
    return os.path.join(dst_dir, name + out_ext)


def prepare_display_copy(src_path, dst_dir=None, max_dim=None):
    """Create (or reuse) a display-sized copy of an image. Returns its path or None.

    Decoding a 12-megapixel JPEG on a Pi takes hundreds of milliseconds; doing
    it once here keeps the render loop's loads to a few milliseconds each.
    """
    dst_path = scaled_path_for(src_path, dst_dir)
    try:
        src_mtime = os.path.getmtime(src_path)
        if os.path.isfile(dst_path) and os.path.getmtime(dst_path) >= src_mtime:
            return dst_path
    except OSError:
        return None

    img = read_image(src_path)
    if img is None:
        return None
    img = downscale_to(img, max_dim or config.IMAGE_MAX_DIM)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    tmp_path = dst_path + ".tmp" + os.path.splitext(dst_path)[1]
    params = [cv2.IMWRITE_JPEG_QUALITY, 90] if dst_path.lower().endswith(".jpg") else []
    try:
        if not cv2.imwrite(tmp_path, img, params):
            raise OSError("imwrite failed")
        os.replace(tmp_path, dst_path)
    except (OSError, cv2.error) as exc:
        print(f"Failed to write display copy for {os.path.basename(src_path)}: {exc}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return None
    return dst_path


def prune_display_copies(valid_sources, dst_dir=None):
    """Delete display copies whose source is gone."""
    dst_dir = dst_dir or config.SCALED_DIR
    if not os.path.isdir(dst_dir):
        return
    keep = {os.path.basename(scaled_path_for(p, dst_dir)) for p in valid_sources}
    for entry in os.listdir(dst_dir):
        if entry in keep:
            continue
        try:
            os.remove(os.path.join(dst_dir, entry))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def to_bgr(image):
    """Convert an image to 3-channel BGR (no copy when already BGR)."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.shape[2] == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def resize_to_fit(image, box_w, box_h):
    """Scale to fit inside box_w x box_h, preserving aspect ratio."""
    h, w = image.shape[:2]
    scale = min(box_w / float(w), box_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    if new_w == w and new_h == h:
        return image
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def paste(background, image, top, left):
    """Paste image (BGR or BGRA) onto background in place, clipping to bounds."""
    bh, bw = background.shape[:2]
    h, w = image.shape[:2]
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(bh, top + h), min(bw, left + w)
    if y1 <= y0 or x1 <= x0:
        return background
    src = image[y0 - top:y1 - top, x0 - left:x1 - left]
    roi = background[y0:y1, x0:x1]

    if src.ndim == 3 and src.shape[2] == 4:
        alpha = src[:, :, 3:4].astype(np.float32) * (1.0 / 255.0)
        rgb = src[:, :, :3].astype(np.float32)
        blended = roi.astype(np.float32) * (1.0 - alpha) + rgb * alpha
        roi[:] = blended.astype(np.uint8)
    else:
        roi[:] = src if src.ndim == 3 else cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    return background


def resize_and_pad(image, width, height):
    """Fit an image inside (width, height) on black. Returns a BGR frame."""
    if image is None:
        return None
    img = to_bgr(image)
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return None
    resized = resize_to_fit(img, width, height)
    padded = np.zeros((height, width, 3), dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    padded[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return padded


def create_zoomed_blurred_background(image, width, height, zoom=1.1):
    """A cover-cropped, zoomed, heavily blurred copy of the image as a backdrop.

    The blur happens at quarter resolution and is scaled back up, which looks
    identical for a background but costs a fraction of a full-size blur.
    """
    if image is None or image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    img = to_bgr(image)
    h, w = img.shape[:2]

    # Region of the source that covers the frame after zooming.
    cover = max(width / float(w), height / float(h)) * zoom
    region_w = max(1, min(w, int(round(width / cover))))
    region_h = max(1, min(h, int(round(height / cover))))
    x0 = (w - region_w) // 2
    y0 = (h - region_h) // 2
    crop = img[y0:y0 + region_h, x0:x0 + region_w]

    small_w, small_h = max(8, width // 4), max(8, height // 4)
    small = cv2.resize(crop, (small_w, small_h), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 4)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


# ---------------------------------------------------------------------------
# Frame builders
# ---------------------------------------------------------------------------

def create_single_image_with_background(image, width, height, margin=0.9):
    """Center the image on a blurred backdrop of itself."""
    if image is None or image.size == 0:
        return np.zeros((height, width, 3), dtype=np.uint8)
    background = create_zoomed_blurred_background(image, width, height)
    fitted = resize_to_fit(image, int(width * margin), int(height * margin))
    top = (height - fitted.shape[0]) // 2
    left = (width - fitted.shape[1]) // 2
    return paste(background, fitted, top, left)


def stitch_images(images, width, height, margin=0.9):
    """Lay several images out in a grid over a blurred backdrop."""
    images = [img for img in images if img is not None and img.size]
    if not images:
        return np.zeros((height, width, 3), dtype=np.uint8)

    background = create_zoomed_blurred_background(random.choice(images), width, height)
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / float(cols)))
    cell_w, cell_h = width // cols, height // rows

    for idx, img in enumerate(images):
        fitted = resize_to_fit(img, int(cell_w * margin), int(cell_h * margin))
        row, col = divmod(idx, cols)
        left = col * cell_w + (cell_w - fitted.shape[1]) // 2
        top = row * cell_h + (cell_h - fitted.shape[0]) // 2
        paste(background, fitted, top, left)
    return background


def normalize_frame_for_display(frame, enforce_size=True):
    """Coerce anything image-like into a BGR uint8 frame (frame-sized if asked)."""
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return None
    normalized = frame
    if normalized.dtype != np.uint8:
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    if normalized.ndim == 2:
        normalized = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
    elif normalized.ndim == 3 and normalized.shape[2] != 3:
        normalized = to_bgr(normalized)
    elif normalized.ndim != 3:
        return None

    if enforce_size and normalized.shape[:2] != (config.FRAME_HEIGHT, config.FRAME_WIDTH):
        normalized = cv2.resize(
            normalized, (config.FRAME_WIDTH, config.FRAME_HEIGHT), interpolation=cv2.INTER_AREA,
        )
    return np.ascontiguousarray(normalized)
