"""Small helpers shared across the package: text cleanup, clocks, JSON caches."""

import json
import os
import re
import unicodedata
from datetime import datetime, timedelta

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - very old Python
    ZoneInfo = None

from slider import config

# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------

_REPLACEMENTS = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "…": "...",
    " ": " ",
}


def sanitize_text(text) -> str:
    """Return ASCII-only text the Hershey fonts can draw.

    Curly quotes and dashes become their ASCII equivalents and accented
    letters are transliterated (e.g. "Émile" -> "Emile") instead of dropped.
    """
    if not text:
        return ""
    text = str(text)
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKD", text)
    return text.encode("ascii", "ignore").decode("ascii")


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------

_TZ = None
_TZ_RESOLVED = False


def local_tz():
    """The configured timezone, or None to mean "system local time"."""
    global _TZ, _TZ_RESOLVED
    if not _TZ_RESOLVED:
        _TZ_RESOLVED = True
        if ZoneInfo is not None and config.TIMEZONE:
            try:
                _TZ = ZoneInfo(config.TIMEZONE)
            except Exception as exc:
                print(f"Unknown timezone '{config.TIMEZONE}' ({exc}); using system local time.")
    return _TZ


def now_local() -> datetime:
    """Current time in the configured timezone."""
    tz = local_tz()
    return datetime.now(tz) if tz is not None else datetime.now()


def from_timestamp(ts) -> datetime:
    """Convert a Unix timestamp to the configured timezone."""
    tz = local_tz()
    return datetime.fromtimestamp(ts, tz) if tz is not None else datetime.fromtimestamp(ts)


def get_central_time() -> datetime:
    """Backwards-compatible alias for now_local()."""
    return now_local()


# ---------------------------------------------------------------------------
# JSON files
# ---------------------------------------------------------------------------

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


def read_json(path):
    """Load a JSON file, returning None if it is missing or unreadable."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(f"Failed to read {os.path.basename(path)}: {exc}")
        return None


def write_json_atomic(path, payload) -> bool:
    """Write JSON via a temp file + rename so a crash never leaves a torn file."""
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        print(f"Failed to write {os.path.basename(path)}: {exc}")
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        return False


def read_timed_cache(path, ttl: timedelta):
    """Read a JSON cache written by write_timed_cache.

    Returns (payload, fresh). payload is None when nothing usable is on disk.
    """
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None, False
    try:
        timestamp = datetime.strptime(payload.get("timestamp", ""), TIMESTAMP_FMT)
    except (TypeError, ValueError):
        return None, False
    age = datetime.now() - timestamp
    return payload, timedelta(0) <= age < ttl


def write_timed_cache(path, payload: dict) -> bool:
    """Write a JSON cache stamped with the current time."""
    stamped = dict(payload)
    stamped["timestamp"] = datetime.now().strftime(TIMESTAMP_FMT)
    return write_json_atomic(path, stamped)


# ---------------------------------------------------------------------------
# Robust JSON extraction from model output
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_CITATION_RE = re.compile(r"\s*\(\[[^\]]*\]\([^)]*\)\)")


def extract_json(text):
    """Pull the first JSON object/array out of free-form model text.

    Handles markdown fences, leading prose, trailing commentary, and the
    inline "([source](url))" citations that web-search responses add.
    Returns the parsed value or None.
    """
    if not text:
        return None
    text = str(text)

    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.append(text)

    for candidate in candidates:
        candidate = candidate.strip()
        # Strip "([outlet](url))" citations first: they can sit inside JSON strings.
        for attempt in (_CITATION_RE.sub("", candidate), candidate):
            parsed = _parse_first_json_value(attempt)
            if parsed is not None:
                return parsed
    return None


def _parse_first_json_value(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char not in "{[":
            continue
        try:
            value, _ = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, (dict, list)):
            return value
    return None
