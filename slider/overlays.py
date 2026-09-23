"""Overlay rendering: status bar (HUD), 5-day forecast, news, and quote panels.

All functions expect a BGR uint8 frame. The panel overlays draw in place and
return the same frame; draw_hud returns a new frame so the underlying slide
can be re-used for the next redraw. Text is wrapped by measured pixel width,
and the quote/news panels shrink their fonts until the content fits.
"""

import cv2
import numpy as np

from slider.image_processing import paste
from slider.touch_ui import draw_mode_buttons
from slider.utils import now_local, sanitize_text
from slider.weather import format_clock, get_weather_icon

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
BAR_COLOR = (50, 50, 50)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def text_size(text, scale, thickness=1):
    """Return (width, height, baseline) of text in pixels."""
    (w, h), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    return w, h, baseline


def line_advance(scale, thickness=1, spacing=1.55):
    """Vertical distance between consecutive text lines at a font scale."""
    _, h, baseline = text_size("Ag", scale, thickness)
    return int(round((h + baseline) * spacing))


def wrap_text(text, scale, max_width, thickness=1):
    """Word-wrap text so no line is wider than max_width pixels.

    Explicit newlines start new lines; blank lines are preserved as "".
    """
    lines = []
    for paragraph in str(text).split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = f"{current} {word}" if current else word
            if text_size(candidate, scale, thickness)[0] <= max_width or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
            # Break words that are wider than the whole line.
            while text_size(current, scale, thickness)[0] > max_width and len(current) > 1:
                cut = len(current) - 1
                while cut > 1 and text_size(current[:cut], scale, thickness)[0] > max_width:
                    cut -= 1
                lines.append(current[:cut])
                current = current[cut:]
        lines.append(current)
    return lines


def put_text(frame, text, x, y, scale, color, thickness=1):
    cv2.putText(frame, text, (int(x), int(y)), FONT, scale, color, thickness, cv2.LINE_AA)


def blend_rect(frame, x0, y0, x1, y1, color, alpha):
    """Blend a solid color rectangle onto the frame in place (only the ROI is touched)."""
    h, w = frame.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    solid = np.empty_like(roi)
    solid[:] = color
    roi[:] = cv2.addWeighted(roi, 1.0 - alpha, solid, alpha, 0.0)


# ---------------------------------------------------------------------------
# Status bar (HUD)
# ---------------------------------------------------------------------------

_status_bar_height = None


def status_bar_height():
    global _status_bar_height
    if _status_bar_height is None:
        _, h_time, base_time = text_size("September 30 2026, 12:00 PM", 0.6)
        _, h_weather, base_weather = text_size("72 F, feels 68, light rain", 0.6)
        _status_bar_height = max(h_time + base_time, h_weather + base_weather) + 16
    return _status_bar_height


def format_conditions(current):
    """One-line summary of current conditions for the status bar."""
    if not current or current.get("temp") is None:
        return "Weather data unavailable"
    temp = round(current["temp"])
    parts = [f"{temp} F"]
    feels = current.get("feels_like")
    if feels is not None and abs(round(feels) - temp) >= 3:
        parts.append(f"feels {round(feels)}")
    description = sanitize_text(current.get("description") or current.get("main") or "")
    if description:
        parts.append(description)
    wind = current.get("wind_speed") or 0
    if wind >= 15:
        parts.append(f"wind {round(wind)} mph")
    return ", ".join(parts)


def draw_hud(frame, hud, now=None):
    """Return a copy of frame with the status bar, mode label, and buttons drawn."""
    now = now or now_local()
    out = frame.copy()
    h, w = out.shape[:2]
    bar_h = status_bar_height()
    blend_rect(out, 0, h - bar_h, w, h, BAR_COLOR, 0.8)

    text_y = h - 12
    conditions = format_conditions(hud.weather)
    put_text(out, conditions, 10, text_y, 0.6, WHITE)
    time_text = now.strftime("%B %d %Y, %I:%M %p")
    time_w = text_size(time_text, 0.6)[0]
    put_text(out, time_text, w - time_w - 10, text_y, 0.6, WHITE)

    if hud.status_text:
        # Centered in the free space of the bar so it never covers panel headers.
        left_edge = text_size(conditions, 0.6)[0] + 30
        right_edge = w - time_w - 30
        status_w = text_size(hud.status_text, 0.5)[0]
        if right_edge - left_edge >= status_w:
            put_text(out, hud.status_text, (left_edge + right_edge - status_w) // 2, text_y, 0.5, (200, 200, 200))

    if hud.show_buttons and hud.buttons:
        draw_mode_buttons(out, hud.buttons, hud.mode)
    return out


# ---------------------------------------------------------------------------
# 5-day forecast overlay
# ---------------------------------------------------------------------------

def add_forecast_overlay(frame, forecast, current=None):
    """Draw the 5-day forecast panel across the top of the frame (in place)."""
    try:
        h, w = frame.shape[:2]
        panel_h = min(232, h - status_bar_height() - 8)
        blend_rect(frame, 0, 0, w, panel_h, BAR_COLOR, 0.7)
        put_text(frame, "5-Day Forecast", 10, 30, 0.8, WHITE, 2)

        if current and current.get("sunrise") and current.get("sunset"):
            sun_text = f"Sunrise {format_clock(current['sunrise'])}   Sunset {format_clock(current['sunset'])}"
            sun_w = text_size(sun_text, 0.45)[0]
            put_text(frame, sun_text, w - sun_w - 12, 28, 0.45, WHITE)

        if not forecast:
            put_text(frame, "Forecast data unavailable", 10, 100, 0.5, WHITE)
            return frame

        days = forecast[:5]
        col_w = w // max(1, len(days))
        icon_size = 56
        for i, day in enumerate(days):
            x = i * col_w + 10
            date_text = str(day.get("date", ""))
            weekday, _, month_day = date_text.partition(",")
            put_text(frame, weekday.strip(), x, 58, 0.55, WHITE, 2)
            put_text(frame, month_day.strip(), x, 80, 0.45, WHITE)

            icon = get_weather_icon(day.get("description", ""), icon_size)
            if icon is not None:
                paste(frame, icon, 90, x + 10)

            try:
                temp_text = f"{round(day['temp_min'])} - {round(day['temp_max'])} F"
            except (KeyError, TypeError, ValueError):
                temp_text = ""
            put_text(frame, temp_text, x, 168, 0.5, WHITE)

            desc_lines = wrap_text(sanitize_text(day.get("description", "")), 0.45, col_w - 16)
            put_text(frame, desc_lines[0] if desc_lines else "", x, 190, 0.45, WHITE)

            pop = day.get("pop") or 0
            if pop >= 0.1:
                kind = "Snow" if "snow" in str(day.get("description", "")).lower() else "Rain"
                put_text(frame, f"{kind} {int(round(pop * 100))}%", x, 212, 0.42, (255, 210, 150))
        return frame
    except Exception as exc:
        print(f"Error adding forecast overlay: {exc}")
        return frame


# ---------------------------------------------------------------------------
# News overlay
# ---------------------------------------------------------------------------

def add_news_overlay(frame, story, fetched_at=None):
    """Draw a news story panel over the frame (in place)."""
    try:
        h, w = frame.shape[:2]
        panel_bottom = max(h - status_bar_height(), 0)
        text_color = (35, 35, 35)
        muted = (110, 110, 110)
        accent = (45, 90, 160)
        title_color = (20, 55, 110)

        blend_rect(frame, 0, 0, w, panel_bottom, (245, 245, 245), 0.85)
        cv2.rectangle(frame, (0, 0), (6, panel_bottom), accent, -1)

        story = story or {}
        headline = sanitize_text(story.get("headline", ""))
        summary = sanitize_text(story.get("summary", ""))
        why = sanitize_text(story.get("why_it_matters", ""))
        sources = [s for s in (story.get("sources") or []) if isinstance(s, str)]
        bias_text = f"Bias: {story.get('bias', 'Center')}"
        if story.get("bias_note"):
            bias_text += f" ({sanitize_text(story['bias_note'])})"

        x = 18
        max_w = w - x - 18
        top = 28
        put_text(frame, "News Update", x, top, 0.62, title_color, 2)

        meta_parts = []
        if story.get("category"):
            meta_parts.append(str(story["category"]))
        if fetched_at is not None:
            meta_parts.append(f"as of {fetched_at.strftime('%I:%M %p').lstrip('0')}")
        if meta_parts:
            meta = "  |  ".join(meta_parts)
            put_text(frame, meta, w - text_size(meta, 0.45)[0] - 16, top, 0.45, muted)
        cv2.line(frame, (x, top + 12), (w - 18, top + 12), (200, 200, 200), 1)

        available = panel_bottom - (top + 34) - 8
        for scale in (1.0, 0.92, 0.85, 0.78, 0.7, 0.62):
            s_head, s_body, s_small = 0.72 * scale, 0.55 * scale, 0.46 * scale
            blocks = [
                (wrap_text(headline, s_head, max_w, 2), s_head, text_color, 2, line_advance(s_head, 2)),
                (wrap_text(summary, s_body, max_w), s_body, text_color, 1, line_advance(s_body)),
            ]
            if why:
                blocks.append((wrap_text(f"Why it matters: {why}", s_small, max_w), s_small, accent, 1, line_advance(s_small)))
            if sources:
                blocks.append((wrap_text("Sources: " + ", ".join(sources[:4]), s_small, max_w), s_small, muted, 1, line_advance(s_small)))
            blocks.append((wrap_text(bias_text, s_small, max_w), s_small, muted, 1, line_advance(s_small)))
            total = sum(len(lines) * adv + 8 for lines, _, _, _, adv in blocks)
            if total <= available:
                break

        y = top + 34
        for lines, scale_used, color, thickness, adv in blocks:
            for line in lines:
                y += adv
                put_text(frame, line, x, y - 4, scale_used, color, thickness)
            y += 8
        return frame
    except Exception as exc:
        print(f"Error adding news overlay: {exc}")
        return frame


# ---------------------------------------------------------------------------
# Quote overlay (also used for AI weather summaries and status messages)
# ---------------------------------------------------------------------------

def add_quote_overlay(frame, quote, source="", title=None, style=None):
    """Draw a centered card with an optional title bar, the text, and its source."""
    try:
        h, w = frame.shape[:2]
        quote = sanitize_text(quote)
        title = sanitize_text(title) if title else ""
        source = sanitize_text(source) if source else ""
        if source.strip().lower() == "today's weather":  # legacy caller convention
            source = ""

        available_h = h - status_bar_height() - 16
        max_box_w = w - 32
        min_box_w = min(600, max_box_w)
        text_color = (105, 105, 105)
        title_color = (230, 230, 230)

        for scale in (1.0, 0.9, 0.8, 0.72, 0.64, 0.56, 0.48):
            s_quote, s_title, s_source = 0.8 * scale, 0.9 * scale, 0.6 * scale
            text_w = max_box_w - 40
            quote_lines = wrap_text(quote, s_quote, text_w)
            title_lines = wrap_text(title, s_title, text_w) if title else []
            source_lines = wrap_text(f"- {source}", s_source, text_w) if source else []
            adv_quote, adv_title, adv_source = line_advance(s_quote), line_advance(s_title), line_advance(s_source)

            title_h = len(title_lines) * adv_title + 16 if title_lines else 0
            body_h = len(quote_lines) * adv_quote + (len(source_lines) * adv_source + 12 if source_lines else 0)
            box_h = title_h + body_h + 40

            widths = [text_size(line, s_quote)[0] for line in quote_lines if line]
            widths += [text_size(line, s_title)[0] for line in title_lines]
            widths += [text_size(line, s_source)[0] for line in source_lines]
            box_w = min(max_box_w, max(min_box_w, (max(widths) if widths else 200) + 40))
            if box_h <= available_h:
                break

        box_x = (w - box_w) // 2
        box_y = max(8, (available_h - box_h) // 2 + 8)
        blend_rect(frame, box_x, box_y, box_x + box_w, box_y + box_h, WHITE, 0.8)

        y = box_y
        if title_lines:
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + title_h), BAR_COLOR, -1)
            y += 8
            for line in title_lines:
                y += adv_title
                put_text(frame, line, (w - text_size(line, s_title)[0]) // 2, y - 6, s_title, title_color)
            y = box_y + title_h

        y += 20
        for line in quote_lines:
            y += adv_quote
            if line:
                put_text(frame, line, (w - text_size(line, s_quote)[0]) // 2, y - 6, s_quote, text_color)

        if source_lines:
            y += 12
            for line in source_lines:
                y += adv_source
                put_text(frame, line, (w - text_size(line, s_source)[0]) // 2, y - 6, s_source, text_color)
        return frame
    except Exception as exc:
        print(f"Error adding quote overlay: {exc}")
        return frame
