"""Slide transitions.

Each transition is a factory: ``render = transition(current, next_)`` does the
per-pair precomputation once and returns ``render(alpha)`` which produces the
frame for progress ``alpha`` in [0, 1]. The slideshow drives ``alpha`` from the
wall clock, so a transition always lasts exactly TRANSITION_TIME no matter how
fast the hardware is. All heavy lifting is in OpenCV (remap, addWeighted,
blendLinear); no per-pixel Python and no per-frame float conversions of whole
frames.
"""

import math

import cv2
import numpy as np

from slider.image_processing import to_bgr


def prepare(current_img, next_img):
    """Return both images as BGR uint8 with identical shapes (next is resized to current)."""
    current_img = to_bgr(current_img)
    next_img = to_bgr(next_img)
    if current_img.dtype != np.uint8:
        current_img = np.clip(current_img, 0, 255).astype(np.uint8)
    if next_img.dtype != np.uint8:
        next_img = np.clip(next_img, 0, 255).astype(np.uint8)
    if next_img.shape[:2] != current_img.shape[:2]:
        h, w = current_img.shape[:2]
        next_img = cv2.resize(next_img, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(current_img), np.ascontiguousarray(next_img)


def _clamp(alpha):
    return 0.0 if alpha < 0.0 else 1.0 if alpha > 1.0 else float(alpha)


def _smoothstep(t):
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------

def fade(current_img, next_img):
    cur, nxt = prepare(current_img, next_img)

    def render(alpha):
        alpha = _clamp(alpha)
        return cv2.addWeighted(cur, 1.0 - alpha, nxt, alpha, 0.0)

    return render


def slide_left(current_img, next_img):
    cur, nxt = prepare(current_img, next_img)
    w = cur.shape[1]

    def render(alpha):
        dx = int(round(w * _clamp(alpha)))
        frame = np.empty_like(cur)
        if dx < w:
            frame[:, :w - dx] = cur[:, dx:]
        if dx > 0:
            frame[:, w - dx:] = nxt[:, :dx]
        return frame

    return render


def slide_right(current_img, next_img):
    cur, nxt = prepare(current_img, next_img)
    w = cur.shape[1]

    def render(alpha):
        dx = int(round(w * _clamp(alpha)))
        frame = np.empty_like(cur)
        if dx < w:
            frame[:, dx:] = cur[:, :w - dx]
        if dx > 0:
            frame[:, :dx] = nxt[:, w - dx:]
        return frame

    return render


def wipe_top(current_img, next_img):
    cur, nxt = prepare(current_img, next_img)
    h = cur.shape[0]

    def render(alpha):
        dy = int(round(h * _clamp(alpha)))
        frame = np.empty_like(cur)
        if dy < h:
            frame[:h - dy] = cur[dy:]
        if dy > 0:
            frame[h - dy:] = nxt[:dy]
        return frame

    return render


def wipe_bottom(current_img, next_img):
    cur, nxt = prepare(current_img, next_img)
    h = cur.shape[0]

    def render(alpha):
        dy = int(round(h * _clamp(alpha)))
        frame = np.empty_like(cur)
        if dy < h:
            frame[dy:] = cur[:h - dy]
        if dy > 0:
            frame[:dy] = nxt[h - dy:]
        return frame

    return render


# ---------------------------------------------------------------------------
# Advanced transitions
# ---------------------------------------------------------------------------

def melt(current_img, next_img):
    """The current image slides downward while fading, revealing the next."""
    cur, nxt = prepare(current_img, next_img)
    h = cur.shape[0]
    max_shift = h // 2

    def render(alpha):
        alpha = _clamp(alpha)
        shift = int(alpha * max_shift)
        frame = nxt.copy()
        if shift < h:
            blended = cv2.addWeighted(cur[:h - shift], 1.0 - alpha, nxt[shift:], alpha, 0.0)
            frame[shift:] = blended
        return frame

    return render


def wave(current_img, next_img):
    """The current image ripples with a sine distortion while fading out."""
    cur, nxt = prepare(current_img, next_img)
    h, w = cur.shape[:2]
    max_vertical = h * 0.4
    max_horizontal = w * 0.02

    ys, xs = np.indices((h, w), dtype=np.float32)
    rows = np.arange(h, dtype=np.float32) / float(h)
    phase_v = rows * (2.0 * np.pi)
    phase_h = rows * (4.0 * np.pi)

    def render(alpha):
        alpha = _clamp(alpha)
        v_shift = np.sin(phase_v + alpha * 2.0 * np.pi) * (max_vertical * alpha)
        h_shift = np.sin(phase_h + alpha * 4.0 * np.pi) * (max_horizontal * alpha)
        map_y = ys + v_shift[:, np.newaxis].astype(np.float32)
        map_x = xs + h_shift[:, np.newaxis].astype(np.float32)
        warped = cv2.remap(cur, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return cv2.addWeighted(warped, 1.0 - alpha, nxt, alpha, 0.0)

    return render


def ripple(current_img, next_img):
    """A soft-edged circle grows from the center, revealing the next image."""
    cur, nxt = prepare(current_img, next_img)
    h, w = cur.shape[:2]
    ys, xs = np.indices((h, w), dtype=np.float32)
    distance = np.sqrt((xs - w / 2.0) ** 2 + (ys - h / 2.0) ** 2).astype(np.float32)
    max_radius = math.hypot(w / 2.0, h / 2.0)
    band = 10.0

    def render(alpha):
        # Sweep the soft band from fully outside the frame (alpha 0) to fully
        # past the corners (alpha 1) so the endpoints are exactly cur and nxt.
        radius = _clamp(alpha) * (max_radius + 2.0 * band) - band
        weight_next = (radius + band - distance) * (1.0 / (2.0 * band))
        np.clip(weight_next, 0.0, 1.0, out=weight_next)
        weight_next = _smoothstep(weight_next)
        return cv2.blendLinear(nxt, cur, weight_next, 1.0 - weight_next)

    return render


def petal_bloom(current_img, next_img):
    """The current image splits into eight petals that rotate and scale away."""
    cur, nxt = prepare(current_img, next_img)
    h, w = cur.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    petals = 8
    max_rotation = math.radians(30)
    scale_factor = 0.3
    blend_boundary = math.radians(2)

    ys, xs = np.indices((h, w), dtype=np.float32)
    dx = xs - cx
    dy = ys - cy
    angle = np.arctan2(dy, dx)
    angle_norm = np.mod(angle + 2.0 * math.pi, 2.0 * math.pi)
    petal_angle = 2.0 * math.pi / petals
    petal_center = np.floor(angle_norm / petal_angle) * petal_angle + petal_angle / 2.0
    angle_diff = np.mod(angle_norm - petal_center + math.pi, 2.0 * math.pi) - math.pi
    half_sign = np.where(angle_diff > 0, 1.0, -1.0).astype(np.float32)

    boundary_dist = np.abs(angle_diff) - (petal_angle / 2.0 - blend_boundary)
    edge = np.clip(boundary_dist / blend_boundary, 0.0, 1.0)
    boundary_mask = np.where(boundary_dist > 0, _smoothstep(1.0 - edge), 1.0).astype(np.float32)
    ones = np.ones((h, w), dtype=np.float32)

    def render(alpha):
        alpha = _clamp(alpha)
        # Petals grow outward (source coordinates shrink) as they rotate and fade.
        k = 1.0 / (1.0 + alpha * scale_factor)
        rot = alpha * max_rotation
        c, s = math.cos(rot), math.sin(rot)
        hs = half_sign * s
        # Rotate each petal about the center by +/- rot and scale by k, using
        # cos(a+r) = cos a cos r - sin a sin r with radius*cos a = dx, radius*sin a = dy.
        map_x = (k * (dx * c - dy * hs) + cx).astype(np.float32)
        map_y = (k * (dy * c + dx * hs) + cy).astype(np.float32)
        warped = cv2.remap(cur, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        inside = cv2.remap(ones, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # Seams between petals open up gradually (no gaps at alpha 0).
        seams = 1.0 - alpha * (1.0 - boundary_mask)
        weight_cur = inside * seams * (1.0 - alpha)
        return cv2.blendLinear(warped, nxt, weight_cur, 1.0 - weight_cur)

    return render


# ---------------------------------------------------------------------------
# Transition sets and helpers
# ---------------------------------------------------------------------------

BASIC_TRANSITIONS = [fade, slide_left, slide_right, wipe_top, wipe_bottom]
ADVANCED_TRANSITIONS = [melt, wave, ripple, petal_bloom]
ALL_TRANSITIONS = BASIC_TRANSITIONS + ADVANCED_TRANSITIONS
# Cheap enough for a Raspberry Pi at full frame rate.
LOW_POWER_TRANSITIONS = BASIC_TRANSITIONS + [melt, wave]


def iter_frames(transition, current_img, next_img, num_frames):
    """Yield num_frames frames of a transition at evenly spaced alphas (0 -> 1)."""
    render = transition(current_img, next_img)
    if num_frames <= 1:
        yield render(1.0)
        return
    for alpha in np.linspace(0.0, 1.0, num_frames):
        yield render(float(alpha))


# Generator-style aliases (current, next, num_frames) kept for compatibility.
def _generator_alias(transition):
    def alias(current_img, next_img, num_frames):
        return iter_frames(transition, current_img, next_img, num_frames)
    alias.__name__ = f"{transition.__name__}_transition"
    return alias


fade_transition = _generator_alias(fade)
slide_transition_left = _generator_alias(slide_left)
slide_transition_right = _generator_alias(slide_right)
wipe_transition_top = _generator_alias(wipe_top)
wipe_transition_bottom = _generator_alias(wipe_bottom)
melt_transition = _generator_alias(melt)
wave_transition = _generator_alias(wave)
zen_ripple_transition = _generator_alias(ripple)
dynamic_petal_bloom_transition = _generator_alias(petal_bloom)
