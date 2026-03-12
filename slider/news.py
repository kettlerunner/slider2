"""AI-generated news retrieval with disk + memory caching."""

import json
import os
import random
from datetime import datetime

from slider import config
from slider.utils import sanitize_text
from slider import ai_client


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------

_news_cache = {
    "expires": datetime.min,
    "pool": [{
        "headline": "News unavailable",
        "summary": "No updates retrieved yet.",
        "sources": [],
        "bias": "Center",
        "bias_note": "",
    }],
    "status": "failure",
}

_FALLBACK_VALUE = {
    "headline": "News unavailable",
    "summary": "Unable to retrieve the latest update.",
    "sources": [],
    "bias": "Center",
    "bias_note": "",
}


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_news_payload(parsed):
    """Validate and normalize a single news story dict."""
    if not isinstance(parsed, dict):
        return None

    headline = sanitize_text(str(parsed.get("headline", "")).strip())
    summary = sanitize_text(str(parsed.get("summary", "")).strip())

    sources = []
    raw_sources = parsed.get("sources", [])
    if isinstance(raw_sources, list):
        for entry in raw_sources:
            name = None
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("source") or entry.get("publisher")
            elif isinstance(entry, str):
                name = entry
            if name:
                cleaned = sanitize_text(str(name).strip())
                if cleaned:
                    sources.append(cleaned)

    bias_label = parsed.get("bias_label") or parsed.get("bias") or "center"
    bias_label = str(bias_label).strip().lower()
    bias_map = {"left": "Left", "center": "Center", "right": "Right"}
    bias = bias_map.get(bias_label, "Center")
    bias_note = sanitize_text(str(parsed.get("bias_note", "")).strip())

    if not headline or not summary:
        return None

    return {
        "headline": headline,
        "summary": summary,
        "sources": sources,
        "bias": bias,
        "bias_note": bias_note,
    }


def _normalize_news_pool(parsed):
    """Normalize a list or dict of news stories into a pool."""
    if isinstance(parsed, dict) and isinstance(parsed.get("stories"), list):
        stories = parsed.get("stories")
    elif isinstance(parsed, list):
        stories = parsed
    elif isinstance(parsed, dict):
        normalized = _normalize_news_payload(parsed)
        return [normalized] if normalized else None
    else:
        return None

    normalized_pool = []
    for entry in stories:
        normalized = _normalize_news_payload(entry)
        if normalized:
            normalized_pool.append(normalized)
        if len(normalized_pool) >= config.NEWS_POOL_SIZE:
            break

    return normalized_pool if normalized_pool else None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _load_news_cache_from_disk():
    """Load news pool from disk cache file."""
    if not os.path.exists(config.NEWS_CACHE_FILE):
        return None
    try:
        with open(config.NEWS_CACHE_FILE, "r") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Failed to read news cache: {exc}")
        return None

    if not isinstance(payload, dict):
        return None

    timestamp_str = payload.get("timestamp")
    try:
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None

    pool = payload.get("pool")
    value = payload.get("value")
    status = payload.get("status", "failure")

    if isinstance(pool, list) and pool:
        normalized_pool = []
        for entry in pool:
            normalized = _normalize_news_payload(entry)
            if normalized:
                normalized_pool.append(normalized)
        if not normalized_pool:
            return None
        pool = normalized_pool
    elif isinstance(value, dict):
        normalized = _normalize_news_payload(value)
        if not normalized:
            return None
        pool = [normalized]
    else:
        return None

    return {
        "timestamp": timestamp,
        "pool": pool,
        "status": status if status in {"success", "failure"} else "failure",
    }


def _save_news_cache_to_disk(pool, status):
    """Write news pool to disk cache file."""
    try:
        first_value = pool[0] if isinstance(pool, list) and pool else None
        with open(config.NEWS_CACHE_FILE, "w") as handle:
            json.dump({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pool": pool,
                "value": first_value,
                "status": status,
            }, handle)
    except OSError as exc:
        print(f"Failed to write news cache: {exc}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_NEWS_PROMPT = """
Use web search to find six current news stories from the last 48 hours.
Focus on: US politics/economics/Wall Street, international relations/trade, or
semiconductor/AI/science/technology policy. Avoid pop culture, celebrity, or
gossip topics. Gather at least 3 similar articles from different outlets so
the summary avoids political bias. Summarize neutrally in 1-2 sentences.

Return JSON ONLY:
{
  "stories": [
    {
      "headline": "A concise headline",
      "summary": "Neutral 1-2 sentence summary",
      "sources": [
        {"name": "Outlet name", "url": "https://example.com/article"}
      ],
      "bias_label": "left|center|right",
      "bias_note": "Short note on why this is the label (e.g., mixed sources)."
    }
  ]
}
""".strip()


def get_ai_generated_news():
    """Retrieve a short AI-generated news blurb. Uses memory + disk caching."""
    now = datetime.now()
    cache_entry = _news_cache

    # 1) Check in-memory cache
    if now < cache_entry["expires"] and cache_entry.get("pool"):
        return random.choice(cache_entry["pool"])

    # 2) Check disk cache
    disk_cache = _load_news_cache_from_disk()
    if disk_cache:
        age = now - disk_cache["timestamp"]
        ttl = config.NEWS_POOL_REFRESH_TTL if disk_cache["status"] == "success" else config.NEWS_CACHE_FAILURE_TTL
        if age < ttl:
            _news_cache.update({
                "expires": now + ttl,
                "pool": disk_cache["pool"],
                "status": disk_cache["status"],
            })
            return random.choice(disk_cache["pool"])

    # 3) No valid cache — fetch from OpenAI
    if not ai_client.is_available():
        _news_cache.update({
            "expires": now + config.NEWS_CACHE_FAILURE_TTL,
            "pool": [_FALLBACK_VALUE],
            "status": "failure",
        })
        return _FALLBACK_VALUE

    news_data = ai_client.responses_create(
        prompt=_NEWS_PROMPT,
        model=config.OPENAI_NEWS_MODEL,
        use_web_search=True,
    )

    if news_data:
        if news_data.startswith("```json"):
            news_data = news_data[len("```json"):].strip()
        if news_data.endswith("```"):
            news_data = news_data[:-len("```")].strip()

        try:
            parsed = json.loads(news_data)
        except json.JSONDecodeError as exc:
            print(f"Failed to parse AI news response: {exc}")
        else:
            normalized_pool = _normalize_news_pool(parsed)
            if normalized_pool:
                _news_cache.update({
                    "expires": now + config.NEWS_POOL_REFRESH_TTL,
                    "pool": normalized_pool,
                    "status": "success",
                })
                _save_news_cache_to_disk(normalized_pool, "success")
                return random.choice(normalized_pool)
            print("AI news response missing required fields. Using fallback.")
    else:
        print("AI news request failed or returned no content. Using fallback.")

    # Fallback: stale disk cache or default
    fallback_pool = [_FALLBACK_VALUE]
    if disk_cache and isinstance(disk_cache.get("pool"), list) and disk_cache["pool"]:
        fallback_pool = disk_cache["pool"]

    _news_cache.update({
        "expires": now + config.NEWS_CACHE_FAILURE_TTL,
        "pool": fallback_pool,
        "status": "failure",
    })
    _save_news_cache_to_disk(fallback_pool, "failure")
    return random.choice(fallback_pool)
