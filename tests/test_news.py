import json
from datetime import datetime, timedelta, timezone

from slider import ai_client, config, news


def _story(headline, **extra):
    base = {
        "headline": headline, "summary": "Something happened. It mattered.",
        "why_it_matters": "It affects prices.", "category": "Economy",
        "published_at": datetime.now(timezone.utc).isoformat(),
        "sources": [{"name": "Reuters", "url": "https://r"}, {"name": "AP", "url": "https://ap"}],
        "bias_label": "center", "bias_note": "mixed outlets",
    }
    base.update(extra)
    return base


def test_normalize_pool_filters_dedupes_and_limits(monkeypatch):
    monkeypatch.setattr(config, "NEWS_POOL_SIZE", 3)
    old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
    parsed = {"stories": [
        _story("One"), _story("One"), _story("Old", published_at=old),
        {"headline": "", "summary": "no headline"}, _story("Two", category="Weird", bias_label="LEFT"),
        _story("Three"), _story("Four"),
    ]}
    pool = news.normalize_pool(parsed)
    assert [s["headline"] for s in pool] == ["One", "Two", "Three"]
    assert pool[1]["category"] == "" and pool[1]["bias"] == "Left"
    assert pool[0]["sources"] == ["Reuters", "AP"]
    assert news.normalize_pool("nonsense") is None


def test_refresh_news_success_writes_disk_cache(isolated_caches, monkeypatch):
    news._state.update({"expires": datetime.min, "pool": [], "status": "failure", "fetched_at": None})
    monkeypatch.setattr(ai_client, "is_available", lambda: True)
    captured = {}

    def fake_create(prompt, model=None, use_web_search=False, json_schema=None, schema_name="x"):
        captured.update(prompt=prompt, schema=json_schema, web=use_web_search)
        return json.dumps({"stories": [_story("Fresh")]})

    monkeypatch.setattr(ai_client, "responses_create", fake_create)
    news.refresh_news()
    pool, fetched_at = news.get_news_pool()
    assert [s["headline"] for s in pool] == ["Fresh"] and fetched_at is not None
    assert captured["web"] and captured["schema"] is news.NEWS_SCHEMA
    assert datetime.now().strftime("%B") in captured["prompt"]  # date-aware prompt

    with open(config.NEWS_CACHE_FILE) as handle:
        saved = json.load(handle)
    assert saved["status"] == "success" and saved["pool"][0]["headline"] == "Fresh"

    # A failure keeps the stale stories instead of blanking the screen.
    monkeypatch.setattr(ai_client, "responses_create", lambda *a, **k: None)
    news.refresh_news(force=True)
    pool, _ = news.get_news_pool()
    assert pool[0]["headline"] == "Fresh" and news._state["status"] == "failure"


def test_pick_story_avoids_repeat():
    pool = [{"headline": "A"}, {"headline": "B"}]
    for _ in range(20):
        assert news.pick_story(pool, avoid_headline="A")["headline"] == "B"
    assert news.pick_story([{"headline": "A"}], avoid_headline="A")["headline"] == "A"
    assert news.pick_story([]) is None
