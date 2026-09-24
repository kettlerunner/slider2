"""AI-written weather blurbs (briefing, poem, haiku, zen master) with caching.

generate() performs network I/O and is meant for the background data thread.
get_cached() is what the render loop reads.
"""

import json
import random
from datetime import datetime

from slider import ai_client, config
from slider.utils import now_local, sanitize_text
from slider.weather import format_clock

STYLE_TITLES = {
    "briefing": "Today's Weather Briefing",
    "poem": "Today's Forecast in Verse",
    "haiku": "Today's Haiku Forecast",
    "zen_master": "Today's Zencast",
}

# Briefing is the most useful, so it comes up twice as often.
_STYLE_WEIGHTS = {"briefing": 2, "poem": 1, "haiku": 1, "zen_master": 1}

_STYLE_INSTRUCTIONS = {
    "briefing": (
        "Write a friendly two-sentence briefing (max 45 words) a family glancing at a photo frame "
        "would find useful: what the day feels like, when conditions change, and one practical tip "
        "(umbrella, jacket, sunscreen, etc.)."
    ),
    "poem": (
        "Turn the forecast into a whimsical rhyming poem of exactly four short lines (max 40 words). "
        "Put each line on its own line."
    ),
    "haiku": "Write the forecast as a single haiku (5-7-5 syllables), one verse per line.",
    "zen_master": "Summarize the weather like a Zen master: calm, direct, a little profound. Max 30 words.",
}

UNAVAILABLE = "Weather summary unavailable."

_cache = {"key": None, "expires": datetime.min, "value": None}


def choose_style(style="random"):
    if style in STYLE_TITLES:
        return style
    styles = list(_STYLE_WEIGHTS)
    return random.choices(styles, weights=[_STYLE_WEIGHTS[s] for s in styles], k=1)[0]


def _fmt_temp(value):
    try:
        return f"{round(float(value))}F"
    except (TypeError, ValueError):
        return "?"


def _fmt_pop(value):
    try:
        pct = int(round(float(value) * 100))
    except (TypeError, ValueError):
        return ""
    return f", {pct}% chance of precipitation" if pct >= 10 else ""


def build_prompt(context, style):
    """Compose the prompt from the weather context built by weather.build_ai_context."""
    now = now_local()
    lines = [
        f"You write short on-screen weather blurbs for a digital photo frame in {context.get('city') or 'town'}.",
        f"Local date and time: {now.strftime('%A, %B %d, %Y, %I:%M %p').replace(' 0', ' ')}.",
    ]
    current = context.get("current")
    if current:
        lines.append(
            f"Right now: {_fmt_temp(current.get('temp'))} (feels like {_fmt_temp(current.get('feels_like'))}), "
            f"{current.get('description', '')}, wind {round(current.get('wind_speed') or 0)} mph, "
            f"humidity {round(current.get('humidity') or 0)}%."
        )
        if current.get("sunrise") and current.get("sunset"):
            lines.append(f"Sunrise {format_clock(current['sunrise'])}, sunset {format_clock(current['sunset'])}.")

    today = context.get("today") or []
    if today:
        lines.append("Coming hours (3-hour steps):")
        for item in today:
            lines.append(
                f"- {item.get('time')}: {_fmt_temp(item.get('temp'))}, {item.get('description')}, "
                f"wind {round(item.get('wind_speed') or 0)} mph{_fmt_pop(item.get('pop'))}"
            )

    five_day = context.get("five_day") or []
    if five_day:
        lines.append("Next days:")
        for day in five_day[:5]:
            lines.append(
                f"- {day.get('date')}: {_fmt_temp(day.get('temp_min'))} to {_fmt_temp(day.get('temp_max'))}, "
                f"{day.get('description')}{_fmt_pop(day.get('pop'))}"
            )

    lines.append("")
    lines.append(_STYLE_INSTRUCTIONS[style])
    lines.append(
        "Rules: mention the most notable change (rain arriving, big temperature swing, strong wind) if there is one. "
        "Use Fahrenheit. Plain ASCII text only: no markdown, no emojis, no headings, no preamble."
    )
    return "\n".join(lines)


def _cache_key(context):
    """Key on the forecast shape, not the minute-to-minute current reading."""
    today = [(i.get("time"), round(i.get("temp") or 0), i.get("description"), round((i.get("pop") or 0) * 10))
             for i in context.get("today") or []]
    days = [(d.get("date_key") or d.get("date"), round(d.get("temp_min") or 0), round(d.get("temp_max") or 0), d.get("description"))
            for d in context.get("five_day") or []]
    return json.dumps([now_local().date().isoformat(), today, days], sort_keys=True)


def generate(context, style="random"):
    """Return (text, style) for the context, calling the model when the cache is stale.

    Returns None when no summary could be produced. Blocking: background use only.
    """
    if not context or not (context.get("today") or context.get("five_day")):
        return None

    key = _cache_key(context)
    now = datetime.now()
    if _cache["key"] == key and now < _cache["expires"]:
        return _cache["value"]

    value = None
    if ai_client.is_available():
        chosen = choose_style(style)
        text = ai_client.responses_create(
            prompt=build_prompt(context, chosen),
            model=config.OPENAI_CHAT_MODEL,
            use_web_search=False,
        )
        if text:
            value = (sanitize_text(text).strip(), chosen)
        else:
            print(f"Error generating forecast summary in {chosen} style.")

    ttl = config.AI_CACHE_SUCCESS_TTL if value else config.AI_CACHE_FAILURE_TTL
    _cache.update({"key": key, "value": value, "expires": now + ttl})
    return value


def get_cached():
    """Return the last generated (text, style) without any network access, or None."""
    return _cache["value"]


def get_or_generate_forecast_summary(weather_data, style="random"):
    """Backwards-compatible wrapper: returns (text, style), never None."""
    context = weather_data if isinstance(weather_data, dict) else {"today": weather_data or []}
    return generate(context, style) or (UNAVAILABLE, choose_style(style))
