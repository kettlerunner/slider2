"""Touchscreen mode buttons — build layout and draw on frames."""

import cv2
import numpy as np

from slider import config


def build_mode_buttons():
    """Create button layout for all defined modes."""
    count = len(config.MODE_DEFINITIONS)
    margin = 10
    gap = 8
    height = 44
    y_pos = 8
    total_width = config.FRAME_WIDTH - (margin * 2) - (gap * (count - 1))
    button_width = max(80, int(total_width / count))

    buttons = []
    x_pos = margin
    for entry in config.MODE_DEFINITIONS:
        rect = (x_pos, y_pos, x_pos + button_width, y_pos + height)
        buttons.append({"mode": entry["mode"], "label": entry["label"], "rect": rect})
        x_pos += button_width + gap
    return buttons


def draw_mode_buttons(frame, buttons, active_mode):
    """Render mode buttons on a frame. Returns the modified frame."""
    if not buttons:
        return frame

    overlay = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1

    for button in buttons:
        x1, y1, x2, y2 = button["rect"]
        is_active = button["mode"] == active_mode
        bg_color = (0, 130, 210) if is_active else (60, 60, 60)
        border_color = (255, 255, 255)
        text_color = (255, 255, 255)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), border_color, 1)

        label = button["label"]
        text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(
            overlay, label, (text_x, text_y),
            font, font_scale, text_color, thickness, cv2.LINE_AA,
        )

    return overlay
