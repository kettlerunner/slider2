"""Slide transition effects — basic and advanced.

All transition generators yield frames blending current_img into next_img.
"""

import math

import cv2
import numpy as np

from slider.image_processing import ensure_same_channels


def _transition_alphas(num_frames):
    """Generate alpha values from 0 to 1 over num_frames."""
    if num_frames <= 1:
        return [1.0]
    return np.linspace(0, 1, num_frames, endpoint=True)


# ---------------------------------------------------------------------------
# Basic transitions
# ---------------------------------------------------------------------------

def fade_transition(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return
    for alpha in _transition_alphas(num_frames):
        blended = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
        yield blended


def slide_transition_left(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return
    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dx = int(width * alpha)
        frame = np.zeros_like(current_img)
        if dx < width:
            frame[:, :width - dx] = current_img[:, dx:]
        if dx > 0:
            frame[:, width - dx:] = next_img[:, :dx]
        yield frame


def slide_transition_right(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return
    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dx = int(width * alpha)
        frame = np.zeros_like(current_img)
        if dx < width:
            frame[:, dx:] = current_img[:, :width - dx]
        if dx > 0:
            frame[:, :dx] = next_img[:, width - dx:]
        yield frame


def wipe_transition_top(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return
    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dy = int(height * alpha)
        frame = np.zeros_like(current_img)
        if dy < height:
            frame[:height - dy, :] = current_img[dy:, :]
        if dy > 0:
            frame[height - dy:, :] = next_img[:dy, :]
        yield frame


def wipe_transition_bottom(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return
    height, width = current_img.shape[:2]
    for alpha in _transition_alphas(num_frames):
        dy = int(height * alpha)
        frame = np.zeros_like(current_img)
        if dy < height:
            frame[dy:, :] = current_img[:height - dy, :]
        if dy > 0:
            frame[:dy, :] = next_img[height - dy:, :]
        yield frame


# ---------------------------------------------------------------------------
# Advanced transitions (vectorized for Pi performance)
# ---------------------------------------------------------------------------

def melt_transition(current_img, next_img, num_frames):
    """Rows of current image slide downward, revealing next image beneath.

    Optimized: uses np.roll + vectorized blending instead of per-row Python loop.
    """
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height = current_img.shape[0]
    max_shift = int(height * 0.5)

    curr_f = current_img.astype(np.float32)
    next_f = next_img.astype(np.float32)

    for alpha in _transition_alphas(num_frames):
        shift = int(alpha * max_shift)
        row_alpha = 1.0 - alpha

        # Shift current image down by 'shift' rows
        shifted = np.roll(curr_f, shift, axis=0)
        # Top rows that wrapped around should be transparent (use next_img)
        if shift > 0:
            shifted[:shift, :, :] = next_f[:shift, :, :]

        # Blend shifted current over next
        frame = next_f * (1.0 - row_alpha) + shifted * row_alpha
        yield np.clip(frame, 0, 255).astype(np.uint8)


def wave_transition(current_img, next_img, num_frames):
    """Current image distorts with a wave effect, revealing next image.

    Optimized: uses cv2.remap instead of per-row Python loop.
    """
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    max_vertical_shift = int(height * 0.4)
    max_horizontal_shift = int(width * 0.02)

    # Precompute coordinate grids
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    row_phases_v = (np.arange(height, dtype=np.float32) / float(height)) * 2 * np.pi
    row_phases_h = (np.arange(height, dtype=np.float32) / float(height)) * 4 * np.pi

    curr_f = current_img.astype(np.float32)
    next_f = next_img.astype(np.float32)

    for alpha in _transition_alphas(num_frames):
        row_alpha = 1.0 - alpha

        # Compute vertical displacement per row
        v_shift = np.sin(row_phases_v + alpha * 2 * np.pi) * max_vertical_shift * alpha
        h_shift = np.sin(row_phases_h + alpha * 4 * np.pi) * max_horizontal_shift * alpha

        # Build displacement maps (broadcast row shifts to full grid)
        map_y = ys + v_shift[:, np.newaxis]
        map_x = xs + h_shift[:, np.newaxis]

        # Remap current image through the distortion
        warped = cv2.remap(
            current_img, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        ).astype(np.float32)

        # Blend warped current over next
        frame = next_f * (1.0 - row_alpha) + warped * row_alpha
        yield np.clip(frame, 0, 255).astype(np.uint8)


def zen_ripple_transition(current_img, next_img, num_frames):
    """Circular ripple expanding from center, revealing next image."""
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    center_x, center_y = width // 2, height // 2
    max_radius = np.sqrt((width / 2.0) ** 2 + (height / 2.0) ** 2)

    def smoothstep(t):
        return 3 * t ** 2 - 2 * t ** 3

    ys, xs = np.indices((height, width))
    distances = np.sqrt((xs - center_x) ** 2 + (ys - center_y) ** 2)

    for alpha in _transition_alphas(num_frames):
        radius = alpha * max_radius
        blend_region = 10
        lower_bound = radius - blend_region
        upper_bound = radius + blend_region

        curr = current_img.astype(np.float32)
        nxt = next_img.astype(np.float32)
        frame = nxt.copy()

        factor = np.zeros_like(distances, dtype=np.float32)
        inside = distances < lower_bound
        outside = distances > upper_bound
        blend_zone = ~inside & ~outside

        factor[inside] = 1.0
        if np.any(blend_zone):
            blend_zone_dist = (distances[blend_zone] - lower_bound) / (upper_bound - lower_bound)
            blend_zone_factor = 1.0 - blend_zone_dist
            blend_zone_factor = smoothstep(blend_zone_factor)
            factor[blend_zone] = blend_zone_factor
        factor[outside] = 0.0

        factor_3c = factor[:, :, np.newaxis]
        blended = curr * (1 - factor_3c) + nxt * factor_3c
        yield blended.astype(np.uint8)


def dynamic_petal_bloom_transition(current_img, next_img, num_frames):
    """Radial petal-shaped burst revealing the next image."""
    current_img, next_img = ensure_same_channels(current_img, next_img)
    if current_img is None or next_img is None:
        return

    height, width = current_img.shape[:2]
    center_x, center_y = width / 2.0, height / 2.0

    N = 8
    max_rotation = math.radians(30)
    scale_factor = 0.3
    inward_factor = 0.2
    blend_boundary = math.radians(2)

    def smoothstep(t):
        return 3 * t ** 2 - 2 * t ** 3

    ys, xs = np.indices((height, width))
    dx = xs - center_x
    dy = ys - center_y
    radius = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    angle_norm = (angle + 2 * math.pi) % (2 * math.pi)

    petal_angle = 2 * math.pi / N
    petal_index = (angle_norm // petal_angle).astype(np.int32)

    petal_center_angle = petal_index * petal_angle + petal_angle / 2.0
    angle_diff = angle_norm - petal_center_angle
    angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

    for alpha in _transition_alphas(num_frames):
        frame = next_img.copy().astype(np.float32)

        radius_factor = 1 - inward_factor + alpha * (inward_factor + scale_factor)
        wave = (math.cos(math.pi * (1 - alpha)) + 1) / 2.0
        adjusted_radius_factor = radius_factor * (0.9 + 0.1 * wave)

        rot = alpha * max_rotation
        top_half = angle_diff > 0
        half_sign = np.ones_like(angle_diff, dtype=np.float32)
        half_sign[~top_half] = -1.0
        new_angle = angle + half_sign * rot
        new_radius = radius * adjusted_radius_factor

        half_petal = petal_angle / 2.0
        boundary_dist = np.abs(angle_diff) - (half_petal - blend_boundary)
        boundary_mask = np.ones_like(angle_diff, dtype=np.float32)
        in_blend_zone = boundary_dist > 0
        if np.any(in_blend_zone):
            blend_norm = boundary_dist[in_blend_zone] / blend_boundary
            blend_norm = np.clip(blend_norm, 0, 1)
            blend_val = smoothstep(1 - blend_norm)
            boundary_mask[in_blend_zone] = blend_val

        src_x = (new_radius * np.cos(new_angle) + center_x).astype(np.float32)
        src_y = (new_radius * np.sin(new_angle) + center_y).astype(np.float32)

        inside = (src_x >= 0) & (src_x < width) & (src_y >= 0) & (src_y < height)

        src_xi = np.clip(np.round(src_x[inside]).astype(np.int32), 0, width - 1)
        src_yi = np.clip(np.round(src_y[inside]).astype(np.int32), 0, height - 1)

        old_pixels = current_img[src_yi, src_xi].astype(np.float32)
        factor = (1 - alpha)
        bm = boundary_mask[inside, np.newaxis]
        final_factor = factor * bm

        base = frame[inside, :3]
        blended_rgb = base * (1 - final_factor) + old_pixels[:, :3] * final_factor
        frame[inside, :3] = blended_rgb

        yield frame.astype(np.uint8)


# ---------------------------------------------------------------------------
# Transition sets for mode selection
# ---------------------------------------------------------------------------

BASIC_TRANSITIONS = [
    fade_transition,
    slide_transition_left,
    slide_transition_right,
    wipe_transition_top,
    wipe_transition_bottom,
]

ADVANCED_TRANSITIONS = [
    melt_transition,
    wave_transition,
    zen_ripple_transition,
    dynamic_petal_bloom_transition,
]

ALL_TRANSITIONS = BASIC_TRANSITIONS + ADVANCED_TRANSITIONS
