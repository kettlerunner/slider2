"""Quote loading and random selection from quotes.json."""

import json
import random

from slider import config

_quotes_cache = None
_bag = []

FALLBACK_QUOTE = ("Every day is a fresh start.", "Unknown")


def load_quotes(quotes_file=None):
    """Load quotes from JSON once; later calls return the cached list."""
    global _quotes_cache
    if _quotes_cache is not None and quotes_file is None:
        return _quotes_cache

    path = quotes_file or config.QUOTES_FILE
    quotes = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            quotes = [q for q in data if isinstance(q, dict) and q.get("quote")]
        else:
            print("quotes.json format is unexpected (expected a list).")
    except FileNotFoundError:
        print(f"No quotes file found at {path}. Using fallback quote.")
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Failed to load quotes from {path}: {exc}")

    if quotes_file is None:
        _quotes_cache = quotes
    return quotes


def get_random_quote(quotes_file=None):
    """Return a (quote, author) tuple, cycling through every quote before repeating."""
    global _bag
    quotes = load_quotes(quotes_file)
    if not quotes:
        return FALLBACK_QUOTE

    if quotes_file is not None:
        entry = random.choice(quotes)
    else:
        if not _bag:
            _bag = list(range(len(quotes)))
            random.shuffle(_bag)
        entry = quotes[_bag.pop()]
    return str(entry.get("quote") or ""), str(entry.get("author") or "Unknown")
