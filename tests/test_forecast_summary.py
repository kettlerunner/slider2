from datetime import datetime

from slider import ai_client, forecast_summary
from slider.demo import demo_five_day, demo_today, demo_weather
from slider.weather import build_ai_context


def test_prompt_includes_conditions_and_style():
    context = build_ai_context(demo_weather(), demo_today(), demo_five_day())
    prompt = forecast_summary.build_prompt(context, "briefing")
    assert "Right now: 72F" in prompt and "Sunrise" in prompt
    assert "Coming hours" in prompt and "Next days" in prompt
    assert "chance of precipitation" in prompt
    assert "two-sentence briefing" in prompt
    assert forecast_summary.choose_style("haiku") == "haiku"
    assert forecast_summary.choose_style("random") in forecast_summary.STYLE_TITLES


def test_generate_caches_and_falls_back(monkeypatch):
    forecast_summary._cache.update({"key": None, "expires": datetime.min, "value": None})
    context = build_ai_context(demo_weather(), demo_today(), demo_five_day())
    calls = []
    monkeypatch.setattr(ai_client, "is_available", lambda: True)
    monkeypatch.setattr(ai_client, "responses_create",
                        lambda **kw: calls.append(kw) or "Sunny now, showers later — bring a jacket.")
    text, style = forecast_summary.generate(context, "poem")
    assert text == "Sunny now, showers later - bring a jacket." and style == "poem"
    assert forecast_summary.generate(context, "poem") == (text, style) and len(calls) == 1
    assert forecast_summary.get_cached() == (text, style)

    assert forecast_summary.generate({"today": [], "five_day": []}) is None
    forecast_summary._cache.update({"key": None, "expires": datetime.min, "value": None})
    monkeypatch.setattr(ai_client, "responses_create", lambda **kw: None)
    assert forecast_summary.generate(context) is None
    assert forecast_summary.get_or_generate_forecast_summary(demo_today())[0] == forecast_summary.UNAVAILABLE
