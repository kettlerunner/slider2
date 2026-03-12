"""Overlay rendering — forecast, news, time/weather status bar, and quotes."""

import textwrap
from datetime import datetime

import cv2
import numpy as np

from slider import config
from slider.image_processing import normalize_frame_for_display
from slider.utils import sanitize_text
from slider.weather import get_weather_icon


# ---------------------------------------------------------------------------
# Status bar height helper
# ---------------------------------------------------------------------------

def _get_status_bar_height():
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    sample_time = datetime.now().strftime("%B %d %Y, %I:%M %p")
    sample_weather = "Temp: 99.9 F, Rain predicted"
    text_size_time, _ = cv2.getTextSize(sample_time, font, font_scale, thickness)
    text_size_weather, _ = cv2.getTextSize(sample_weather, font, font_scale, thickness)
    return max(text_size_time[1], text_size_weather[1]) + 20


# ---------------------------------------------------------------------------
# 5-day forecast overlay
# ---------------------------------------------------------------------------

def add_forecast_overlay(frame, forecast):
    """Render the 5-day forecast on a frame."""
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 224), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "5-Day Forecast", (10, 30), font, 0.8, font_color, 2, cv2.LINE_AA)

        if forecast:
            col_width = frame.shape[1] // 5
            for i, day in enumerate(forecast):
                x = i * col_width + 10
                y = 60

                cv2.putText(frame, day["date"].split(",")[0], (x, y),
                            font, font_scale, font_color, thickness, cv2.LINE_AA)
                y += 25
                cv2.putText(frame, day["date"].split(",")[1].strip(), (x, y),
                            font, font_scale, font_color, thickness, cv2.LINE_AA)
                y += 35

                temp_text = f"{day['temp_min']:.1f} - {day['temp_max']:.1f} F"
                cv2.putText(frame, temp_text, (x, y),
                            font, font_scale, font_color, thickness, cv2.LINE_AA)
                y += 35

                icon_img = get_weather_icon(day["description"])
                if icon_img is not None:
                    icon_size = 64
                    icon_img = cv2.resize(icon_img, (icon_size, icon_size))
                    icon_y_offset = y - 20
                    icon_x_offset = x + 20

                    if (0 <= icon_y_offset < frame.shape[0] - icon_size
                            and 0 <= icon_x_offset < frame.shape[1] - icon_size):
                        if icon_img.ndim == 3 and icon_img.shape[2] == 4:
                            alpha = icon_img[:, :, 3] / 255.0
                            alpha = alpha[:, :, np.newaxis]
                            for c in range(3):
                                frame[
                                    icon_y_offset:icon_y_offset + icon_size,
                                    icon_x_offset:icon_x_offset + icon_size,
                                    c,
                                ] = (
                                    icon_img[:, :, c] * alpha[:, :, 0]
                                    + frame[
                                        icon_y_offset:icon_y_offset + icon_size,
                                        icon_x_offset:icon_x_offset + icon_size,
                                        c,
                                    ] * (1 - alpha[:, :, 0])
                                )
                        else:
                            frame[
                                icon_y_offset:icon_y_offset + icon_size,
                                icon_x_offset:icon_x_offset + icon_size,
                            ] = icon_img

                    y += 64

                desc = day["description"]
                cv2.putText(frame, desc, (x, y),
                            font, font_scale * 0.9, font_color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Forecast data unavailable", (10, 100),
                        font, font_scale, font_color, thickness, cv2.LINE_AA)

        return frame
    except Exception as e:
        print(f"Error adding forecast overlay: {e}")
        return frame


# ---------------------------------------------------------------------------
# News overlay
# ---------------------------------------------------------------------------

def add_news_overlay(frame, news):
    """Render a news article overlay on a frame."""
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.62
        font_color = (35, 35, 35)
        thickness = 1
        accent_color = (45, 90, 160)
        title_color = (20, 55, 110)
        panel_color = (245, 245, 245)

        overlay = frame.copy()
        bar_height = _get_status_bar_height()
        y_start = 0
        panel_bottom = max(frame.shape[0] - bar_height, 0)
        cv2.rectangle(overlay, (0, y_start), (frame.shape[1], panel_bottom), panel_color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (0, y_start), (6, panel_bottom), accent_color, -1)

        headline = textwrap.wrap(news.get("headline", ""), width=60)
        summary = textwrap.wrap(news.get("summary", ""), width=60)
        sources = news.get("sources", []) or []
        if isinstance(sources, list):
            sources_line = ", ".join(sources[:4])
        else:
            sources_line = ""
        bias_label = news.get("bias", "Center")
        bias_note = news.get("bias_note", "")
        bias_text = f"Bias: {bias_label}"
        if bias_note:
            bias_text = f"{bias_text} ({bias_note})"
        y = y_start + 28

        cv2.putText(frame, "News Update:", (18, y),
                    font, font_scale, title_color, thickness + 1, cv2.LINE_AA)
        y += 12
        cv2.line(frame, (18, y), (frame.shape[1] - 18, y), (200, 200, 200), 1)
        y += 22

        for line in headline:
            cv2.putText(frame, line, (18, y),
                        font, font_scale, font_color, thickness + 1, cv2.LINE_AA)
            y += 30

        for line in summary:
            cv2.putText(frame, line, (18, y),
                        font, font_scale * 0.85, font_color, thickness, cv2.LINE_AA)
            y += 25

        if sources_line:
            for line in textwrap.wrap(f"Sources: {sources_line}", width=70):
                cv2.putText(frame, line, (18, y),
                            font, font_scale * 0.75, font_color, thickness, cv2.LINE_AA)
                y += 22

        if bias_text:
            for line in textwrap.wrap(bias_text, width=70):
                cv2.putText(frame, line, (18, y),
                            font, font_scale * 0.75, font_color, thickness, cv2.LINE_AA)
                y += 22

        return frame
    except Exception as e:
        print(f"Error adding news overlay: {e}")
        return frame


# ---------------------------------------------------------------------------
# Time + weather status bar
# ---------------------------------------------------------------------------

def add_time_overlay(frame, temp, weather, status_text=None):
    """Render time and weather info in a bottom status bar."""
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None

        time_text = datetime.now().strftime("%B %d %Y, %I:%M %p")
        if temp is not None:
            weather_text = f"Temp: {temp:.1f} F"
        else:
            weather_text = "Weather data unavailable"

        if weather:
            w = weather.lower()
            if "rain" in w:
                weather_text += ", Raining" if "drizzle" in w else ", Rain predicted"
            elif "snow" in w:
                weather_text += ", Snowing" if "flurries" in w else ", Snow predicted"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 1

        text_size_time, _ = cv2.getTextSize(time_text, font, font_scale, thickness)
        text_y = frame.shape[0] - 12
        text_x_weather = 10
        text_x_time = frame.shape[1] - text_size_time[0] - 10

        overlay_frame = frame.copy()
        bar_height = _get_status_bar_height()
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - bar_height),
                      (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)

        cv2.putText(overlay_frame, weather_text, (text_x_weather, text_y),
                    font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(overlay_frame, time_text, (text_x_time, text_y),
                    font, font_scale, font_color, thickness, cv2.LINE_AA)

        if status_text:
            status_font_scale = 0.5
            status_thickness = 1
            status_size, _ = cv2.getTextSize(status_text, font, status_font_scale, status_thickness)
            status_position = (frame.shape[1] - status_size[0] - 10, 30)
            cv2.putText(overlay_frame, status_text,
                        (status_position[0] + 1, status_position[1] + 1),
                        font, status_font_scale, (0, 0, 0), status_thickness, cv2.LINE_AA)
            cv2.putText(overlay_frame, status_text, status_position,
                        font, status_font_scale, font_color, status_thickness, cv2.LINE_AA)

        return overlay_frame
    except Exception as e:
        print(f"Error adding overlay: {e}")
        return frame


# ---------------------------------------------------------------------------
# Quote overlay
# ---------------------------------------------------------------------------

def add_quote_overlay(frame, quote, source="", title=None, style=None):
    """Render a motivational quote in a centered overlay box."""
    try:
        frame = normalize_frame_for_display(frame, enforce_size=False)
        if frame is None:
            return None
        MIN_BOX_WIDTH = 600

        quote = sanitize_text(quote)
        if title:
            title = sanitize_text(title)
        source = sanitize_text(source) if source else ""
        style = sanitize_text(style) if style else ""

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_quote = 0.8
        font_scale_source = 0.6
        font_scale_title = 0.9

        font_color = (105, 105, 105)
        font_color_title = (230, 230, 230)
        box_color = (255, 255, 255)
        title_bar_color = (50, 50, 50)
        thickness = 1

        raw_quote_lines = quote.split("\n")
        quote_lines = []
        for raw_line in raw_quote_lines:
            if raw_line.strip():
                wrapped = textwrap.wrap(raw_line.strip(), width=50)
            else:
                wrapped = [""]
            quote_lines.extend(wrapped)

        title_lines = textwrap.wrap(title, width=50) if title else []

        if source.strip().lower() == "today's weather":
            source_lines = []
        else:
            source_lines = textwrap.wrap(f"- {source}", width=50) if source else []

        line_height_quote = cv2.getTextSize("Test", font, font_scale_quote * 1.3, thickness)[0][1]
        line_height_title = cv2.getTextSize("Test", font, font_scale_title * 1.3, thickness)[0][1]
        line_height_source = (
            cv2.getTextSize("Test", font, font_scale_source * 1.3, thickness)[0][1]
            if source_lines else 0
        )

        text_height = 0
        title_height = 0
        if title_lines:
            title_height = line_height_title * len(title_lines) + 20
            text_height += title_height

        text_height += line_height_quote * len(quote_lines)
        if source_lines:
            text_height += 20 + (line_height_source * len(source_lines))

        max_line_widths = []
        if title_lines:
            for line in title_lines:
                max_line_widths.append(cv2.getTextSize(line, font, font_scale_title, thickness)[0][0])
        for line in quote_lines:
            if line:
                max_line_widths.append(cv2.getTextSize(line, font, font_scale_quote, thickness)[0][0])
        if source_lines:
            for line in source_lines:
                max_line_widths.append(cv2.getTextSize(line, font, font_scale_source, thickness)[0][0])

        calculated_width = max(max_line_widths) if max_line_widths else 200
        box_width = max(calculated_width + 40, MIN_BOX_WIDTH)
        box_height = text_height + 80
        box_x = (frame.shape[1] - box_width) // 2
        box_y = (frame.shape[0] - box_height) // 2

        overlay_frame = frame.copy()
        cv2.rectangle(overlay_frame, (box_x, box_y),
                      (box_x + box_width, box_y + box_height), box_color, -1)
        alpha = 0.8
        cv2.addWeighted(overlay_frame, alpha, frame, 1 - alpha, 0, overlay_frame)

        if title_lines:
            title_bar_y_end = box_y + title_height
            cv2.rectangle(overlay_frame, (box_x, box_y),
                          (box_x + box_width, title_bar_y_end), title_bar_color, -1)

        title_line_sizes = []
        for line in title_lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale_title, thickness)
            title_line_sizes.append(text_size)

        if title_lines:
            total_title_text_height = sum(t[1] for t in title_line_sizes) + (
                (len(title_lines) - 1) * 10
            )
            available_space = title_height - 20
            vertical_offset = (available_space - total_title_text_height) // 2
            y = box_y + 10 + vertical_offset
        else:
            y = box_y + 20

        for i, line in enumerate(title_lines):
            text_size = title_line_sizes[i]
            line_height = text_size[1]
            x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(overlay_frame, line, (x, y + line_height),
                        font, font_scale_title, font_color_title, thickness, cv2.LINE_AA)
            y += line_height
            if i < len(title_lines) - 1:
                y += 10

        if title_lines:
            y = box_y + title_height
        y += 20

        for line in quote_lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale_quote, thickness)
            x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(overlay_frame, line, (x, y + text_size[1]),
                        font, font_scale_quote, font_color, thickness, cv2.LINE_AA)
            y += text_size[1] + 10

        if source_lines:
            y += 10
            for line in source_lines:
                text_size, _ = cv2.getTextSize(line, font, font_scale_source, thickness)
                x = (frame.shape[1] - text_size[0]) // 2
                cv2.putText(overlay_frame, line, (x, y + text_size[1]),
                            font, font_scale_source, font_color, thickness, cv2.LINE_AA)
                y += text_size[1] + 10

        return overlay_frame
    except Exception as e:
        print(f"Error adding quote overlay: {e}")
        return frame
