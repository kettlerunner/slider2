"""Quote loading and random selection from quotes.json."""

import json
import random

from slider import config

_quotes_cache = None


def load_quotes(quotes_file=None):
    """Load quotes from JSON file, caching for subsequent calls."""
    global _quotes_cache
    if _quotes_cache is not None:
        return _quotes_cache

    if quotes_file is None:
        quotes_file = config.resource_path("quotes.json")

    try:
        with open(quotes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            _quotes_cache = data
        else:
            print("quotes.json format is unexpected (expected list).")
            _quotes_cache = []
    except FileNotFoundError:
        print(f"No quotes.json found at {quotes_file}. Using fallback quote.")
        _quotes_cache = []
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Failed to load quotes from {quotes_file}: {exc}")
        _quotes_cache = []

    return _quotes_cache


def get_random_quote(quotes_file=None):
    """Return a random (quote, author) tuple."""
    quotes_list = load_quotes(quotes_file)
    if not quotes_list:
        return "Every day is a fresh start.", "Unknown"

    quote_data = random.choice(quotes_list)
    quote = quote_data.get("quote", "") or ""
    source = quote_data.get("author", "") or ""
    return quote, source
