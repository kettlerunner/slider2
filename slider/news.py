"""AI-generated news digest with memory + disk caching.

The digest is fetched by the background data thread via refresh_news(); the
render loop only ever reads the cached pool through get_news_pool() and
pick_story(), so a slow model call never stalls the screen.
"""

import random
from datetime import datetime, timedelta, timezone

from slider import ai_client, config
from slider.utils import (
    TIMESTAMP_FMT,
    extract_json,
    now_local,
    read_json,
    sanitize_text,
    write_json_atomic,
)

CATEGORIES = ["Politics", "Economy", "World", "Technology", "Science", "Health", "Local"]

FALLBACK_STORY = {
    "headline": "News unavailable",
    "summary": "Unable to retrieve the latest update.",
    "why_it_matters": "",
    "category": "",
    "published_at": "",
    "sources": [],
    "bias": "Center",
    "bias_note": "",
}

_state = {
    "expires": datetime.min,
    "pool": [],
    "status": "failure",
    "fetched_at": None,  # datetime of the last successful digest
}

# ---------------------------------------------------------------------------
# Structured output schema (Responses API json_schema, strict mode)
# ---------------------------------------------------------------------------

_SOURCE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"name": {"type": "string"}, "url": {"type": "string"}},
    "required": ["name", "url"],
}

NEWS_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "stories": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "headline": {"type": "string"},
                    "summary": {"type": "string"},
                    "why_it_matters": {"type": "string"},
                    "category": {"type": "string", "enum": CATEGORIES},
                    "published_at": {"type": "string"},
                    "sources": {"type": "array", "items": _SOURCE_SCHEMA},
                    "bias_label": {"type": "string", "enum": ["left", "center", "right"]},
                    "bias_note": {"type": "string"},
                },
                "required": [
                    "headline", "summary", "why_it_matters", "category",
                    "published_at", "sources", "bias_label", "bias_note",
                ],
            },
        }
    },
    "required": ["stories"],
}


def build_prompt(now=None):
    """Build the date-aware news prompt."""
    now = now or now_local()
    topics = "; ".join(str(t) for t in config.NEWS_TOPICS if t)
    local_line = ""
    if config.NEWS_LOCAL_AREA:
        local_line = f"Include exactly one story about {config.NEWS_LOCAL_AREA} with category \"Local\".\n"
    return (
        f"Today is {now.strftime('%A, %B %d, %Y')} ({now.strftime('%I:%M %p').lstrip('0')} {now.tzname() or 'local time'}).\n"
        f"Use web search to find the {config.NEWS_POOL_SIZE} most significant news stories published in the last "
        f"{config.NEWS_MAX_AGE_HOURS} hours.\n"
        f"Focus on: {topics}.\n"
        f"{local_line}"
        "Skip pop culture, celebrity, sports, and gossip. Each story must be corroborated by at least two "
        "different outlets; list up to three outlets as sources with their article URLs.\n"
        "For each story write:\n"
        "- headline: concise, under 12 words, no clickbait\n"
        "- summary: neutral, factual, 2 sentences, under 55 words, no opinion\n"
        "- why_it_matters: one plain-language sentence under 25 words on why a regular household should care\n"
        "- category: one of Politics, Economy, World, Technology, Science, Health, Local\n"
        "- published_at: ISO 8601 date-time of the earliest source article\n"
        "- bias_label: left, center, or right, reflecting the overall lean of the coverage you used\n"
        "- bias_note: under 15 words explaining the label (e.g. \"mixed outlets, factual reporting\")\n"
        "Use plain ASCII text. Return JSON only, matching the requested schema."
    )


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _parse_published(value):
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def normalize_story(parsed):
    """Validate and clean a single story dict. Returns None if unusable."""
    if not isinstance(parsed, dict):
        return None

    headline = sanitize_text(str(parsed.get("headline", "")).strip())
    summary = sanitize_text(str(parsed.get("summary", "")).strip())
    if not headline or not summary:
        return None

    sources = []
    raw_sources = parsed.get("sources", [])
    if isinstance(raw_sources, list):
        for entry in raw_sources:
            name = None
            if isinstance(entry, dict):
                name = entry.get("name") or entry.get("source") or entry.get("publisher")
            elif isinstance(entry, str):
                name = entry
            cleaned = sanitize_text(str(name).strip()) if name else ""
            if cleaned and cleaned not in sources:
                sources.append(cleaned)

    bias_label = str(parsed.get("bias_label") or parsed.get("bias") or "center").strip().lower()
    bias = {"left": "Left", "center": "Center", "right": "Right"}.get(bias_label, "Center")

    category = sanitize_text(str(parsed.get("category", "")).strip()).title()
    if category not in CATEGORIES:
        category = ""

    published = _parse_published(parsed.get("published_at"))

    return {
        "headline": headline,
        "summary": summary,
        "why_it_matters": sanitize_text(str(parsed.get("why_it_matters", "")).strip()),
        "category": category,
        "published_at": published.isoformat() if published else "",
        "sources": sources[:4],
        "bias": bias,
        "bias_note": sanitize_text(str(parsed.get("bias_note", "")).strip()),
    }


def normalize_pool(parsed, now=None):
    """Turn a parsed model response into a list of stories (or None)."""
    if isinstance(parsed, dict) and isinstance(parsed.get("stories"), list):
        stories = parsed["stories"]
    elif isinstance(parsed, list):
        stories = parsed
    elif isinstance(parsed, dict):
        stories = [parsed]
    else:
        return None

    now = now or datetime.now(timezone.utc)
    max_age = timedelta(hours=config.NEWS_MAX_AGE_HOURS * 1.5)  # lenient: models misdate
    pool = []
    seen = set()
    for entry in stories:
        story = normalize_story(entry)
        if not story:
            continue
        published = _parse_published(story["published_at"])
        if published and now - published > max_age:
            continue
        key = story["headline"].lower()
        if key in seen:
            continue
        seen.add(key)
        pool.append(story)
        if len(pool) >= config.NEWS_POOL_SIZE:
            break
    return pool or None


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _load_disk_cache():
    payload = read_json(config.NEWS_CACHE_FILE)
    if not isinstance(payload, dict):
        return None
    try:
        timestamp = datetime.strptime(payload.get("timestamp", ""), TIMESTAMP_FMT)
    except (TypeError, ValueError):
        return None

    raw_pool = payload.get("pool")
    if not isinstance(raw_pool, list) and isinstance(payload.get("value"), dict):
        raw_pool = [payload["value"]]
    pool = [s for s in (normalize_story(e) for e in (raw_pool or [])) if s]
    if not pool:
        return None

    fetched_at = None
    try:
        fetched_at = datetime.strptime(payload.get("fetched_at", ""), TIMESTAMP_FMT)
    except (TypeError, ValueError):
        fetched_at = timestamp

    status = payload.get("status")
    return {
        "timestamp": timestamp,
        "fetched_at": fetched_at,
        "pool": pool,
        "status": status if status in {"success", "failure"} else "failure",
    }


def _save_disk_cache(pool, status, fetched_at):
    write_json_atomic(config.NEWS_CACHE_FILE, {
        "timestamp": datetime.now().strftime(TIMESTAMP_FMT),
        "fetched_at": fetched_at.strftime(TIMESTAMP_FMT) if fetched_at else "",
        "pool": pool,
        "status": status,
    })


def _adopt(pool, status, ttl, fetched_at, now):
    _state.update({
        "expires": now + ttl,
        "pool": list(pool),
        "status": status,
        "fetched_at": fetched_at,
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refresh_news(force=False):
    """Refresh the news pool if the caches have expired. Safe to call often.

    Performs network I/O; intended for the background data thread.
    """
    now = datetime.now()
    if not force and now < _state["expires"] and _state["pool"]:
        return

    disk = _load_disk_cache()
    if disk and not force:
        ttl = config.NEWS_POOL_REFRESH_TTL if disk["status"] == "success" else config.NEWS_CACHE_FAILURE_TTL
        if now - disk["timestamp"] < ttl:
            _adopt(disk["pool"], disk["status"], ttl - (now - disk["timestamp"]), disk["fetched_at"], now)
            return

    stale_pool = disk["pool"] if disk else list(_state["pool"])
    stale_fetched = disk["fetched_at"] if disk else _state["fetched_at"]

    if not ai_client.is_available():
        _adopt(stale_pool, "failure", config.NEWS_CACHE_FAILURE_TTL, stale_fetched, now)
        return

    text = ai_client.responses_create(
        prompt=build_prompt(),
        model=config.OPENAI_NEWS_MODEL,
        use_web_search=True,
        json_schema=NEWS_SCHEMA,
        schema_name="news_digest",
    )
    pool = normalize_pool(extract_json(text)) if text else None
    if pool:
        _adopt(pool, "success", config.NEWS_POOL_REFRESH_TTL, now, now)
        _save_disk_cache(pool, "success", now)
        print(f"News digest refreshed: {len(pool)} stories.")
        return

    print("AI news request failed or returned nothing usable; keeping previous stories.")
    _adopt(stale_pool, "failure", config.NEWS_CACHE_FAILURE_TTL, stale_fetched, now)
    if stale_pool:
        _save_disk_cache(stale_pool, "failure", stale_fetched)


def get_news_pool():
    """Return (stories, fetched_at) from memory without touching the network."""
    return list(_state["pool"]), _state["fetched_at"]


def pick_story(pool, avoid_headline=None):
    """Pick a story, avoiding an immediate repeat when possible."""
    if not pool:
        return None
    choices = [s for s in pool if s.get("headline") != avoid_headline] or list(pool)
    return random.choice(choices)


def get_ai_generated_news():
    """Refresh if needed and return one story (never None). Blocking."""
    refresh_news()
    return pick_story(_state["pool"]) or FALLBACK_STORY
