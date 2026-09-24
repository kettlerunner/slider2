from datetime import timedelta

from slider.utils import extract_json, read_timed_cache, sanitize_text, write_timed_cache


def test_sanitize_transliterates_accents():
    assert sanitize_text("Émile Zola — “Non”") == 'Emile Zola - "Non"'


def test_sanitize_handles_none_and_ellipsis():
    assert sanitize_text(None) == ""
    assert sanitize_text("Wait…") == "Wait..."


def test_extract_json_from_fenced_block():
    text = 'Here you go:\n```json\n{"stories": [{"headline": "A"}]}\n```\nEnjoy.'
    assert extract_json(text) == {"stories": [{"headline": "A"}]}


def test_extract_json_with_prose_and_citations():
    text = ('Based on my search ([reuters.com](https://reuters.com/x)) the digest is '
            '{"stories": [{"headline": "B ([ap.org](https://ap.org/y))"}]} hope that helps')
    parsed = extract_json(text)
    assert parsed["stories"][0]["headline"] == "B"


def test_extract_json_array_and_garbage():
    assert extract_json('[{"a": 1}, {"a": 2}]') == [{"a": 1}, {"a": 2}]
    assert extract_json("no json here {oops") is None
    assert extract_json("") is None


def test_timed_cache_roundtrip(tmp_path):
    path = str(tmp_path / "cache.json")
    assert read_timed_cache(path, timedelta(minutes=5)) == (None, False)
    assert write_timed_cache(path, {"value": 42})
    payload, fresh = read_timed_cache(path, timedelta(minutes=5))
    assert fresh and payload["value"] == 42
    payload, fresh = read_timed_cache(path, timedelta(seconds=-1))
    assert payload["value"] == 42 and not fresh
