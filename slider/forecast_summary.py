"""AI-generated weather summary (poem, haiku, zen master style) with caching."""

import json
import random
from datetime import datetime

from slider import config
from slider import ai_client
from slider.utils import sanitize_text


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

_forecast_summary_cache = {
    "key": None,
    "expires": datetime.min,
    "value": ("Weather summary unavailable.", "random"),
}

STYLE_TITLES = {
    "poem": "Today's Forecast in Verse",
    "haiku": "Today's Haiku Forecast",
    "zen_master": "Today's Zencast",
}


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def get_tldr_forecast(weather_data, style="random"):
    """Generate a concise forecast summary using OpenAI.

    Returns (summary_text, style_used, success_bool).
    """
    if not weather_data:
        return "Weather summary unavailable.", style, False

    formatted_data = "\n".join(
        f"Time: {item['time']}, Temp: {item['temp']}\u00b0F, Description: {item['description']}, "
        f"Wind Speed: {item['wind_speed']} mph, Humidity: {item['humidity']}%"
        for item in weather_data
    )

    styles = ["poem", "haiku", "zen_master"]
    chosen_style = random.choice(styles) if style == "random" else style

    style_prompts = {
        "poem": "Turn the weather forecast into a very short whimsical poem. Be concise.",
        "haiku": "Write the weather forecast as a tiny haiku. Minimal and poetic.",
        "zen_master": "Summarize the weather like a Zen master. Very concise, direct, profound.",
    }

    prompt = f"""
Here is the weather forecast for today:

{formatted_data}

{style_prompts.get(chosen_style, "Summarize this into a conversational, super short forecast.")}
    """.strip()

    fallback_message = "Weather summary unavailable."

    if not ai_client.is_available():
        return fallback_message, chosen_style, False

    text = ai_client.responses_create(
        prompt=prompt,
        model=config.OPENAI_CHAT_MODEL,
        use_web_search=False,
    )

    if text:
        return sanitize_text(text), chosen_style, True

    print(f"Error generating forecast summary in {chosen_style} style. Using fallback.")
    return fallback_message, chosen_style, False


def get_or_generate_forecast_summary(weather_data, style="random"):
    """Cached wrapper around get_tldr_forecast.

    Returns (summary_text, style_used).
    """
    if not weather_data:
        return "Weather summary unavailable.", "random"

    cache_key = json.dumps(weather_data, sort_keys=True)
    now = datetime.now()
    cache = _forecast_summary_cache

    if cache["key"] == cache_key and now < cache["expires"]:
        return cache["value"]

    summary, style_used, success = get_tldr_forecast(weather_data, style=style)
    ttl = config.AI_CACHE_SUCCESS_TTL if success else config.AI_CACHE_FAILURE_TTL

    cache.update({
        "key": cache_key,
        "value": (summary, style_used),
        "expires": now + ttl,
    })
    return summary, style_used
