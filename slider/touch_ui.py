"""Touchscreen mode buttons: layout, drawing, and hit testing."""

import cv2

from slider import config

FONT = cv2.FONT_HERSHEY_SIMPLEX


def build_mode_buttons():
    """Create the button layout for all configured modes."""
    modes = [m for m in config.MODE_DEFINITIONS if isinstance(m, dict) and m.get("mode")]
    count = len(modes)
    if count == 0:
        return []
    margin, gap, height, y_pos = 10, 8, 44, 8
    total_width = config.FRAME_WIDTH - (margin * 2) - (gap * (count - 1))
    button_width = max(80, int(total_width / count))

    buttons = []
    x_pos = margin
    for entry in modes:
        rect = (x_pos, y_pos, x_pos + button_width, y_pos + height)
        buttons.append({"mode": entry["mode"], "label": str(entry.get("label") or entry["mode"].title()), "rect": rect})
        x_pos += button_width + gap
    return buttons


def draw_mode_buttons(frame, buttons, active_mode):
    """Draw the buttons onto frame in place and return it."""
    for button in buttons:
        x1, y1, x2, y2 = button["rect"]
        bg_color = (0, 130, 210) if button["mode"] == active_mode else (60, 60, 60)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        label = button["label"]
        (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
        cv2.putText(frame, label, (x1 + (x2 - x1 - tw) // 2, y1 + (y2 - y1 + th) // 2),
                    FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def hit_test(buttons, x, y):
    """Return the mode of the button under (x, y), or None."""
    for button in buttons:
        x1, y1, x2, y2 = button["rect"]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return button["mode"]
    return None


def mode_label(mode):
    for entry in config.MODE_DEFINITIONS:
        if isinstance(entry, dict) and entry.get("mode") == mode:
            return str(entry.get("label") or mode.title())
    return str(mode).title()
