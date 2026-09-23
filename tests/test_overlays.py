import numpy as np

from slider import config, overlays
from slider.demo import demo_five_day, demo_news, demo_weather
from slider.display import HudState
from slider.touch_ui import build_mode_buttons, hit_test


def test_wrap_text_respects_pixel_width():
    text = "The quick brown fox jumps over the lazy dog " * 6 + "\n\nSupercalifragilisticexpialidocious" * 3
    lines = overlays.wrap_text(text, 0.8, 300)
    assert all(overlays.text_size(line, 0.8)[0] <= 300 for line in lines)
    assert "" in lines  # blank paragraph preserved


def test_hud_draws_without_modifying_input(frame):
    hud = HudState(build_mode_buttons())
    hud.weather = demo_weather()
    hud.status_text = "Mode: Random"
    hud.show_buttons = True
    before = frame.copy()
    out = overlays.draw_hud(frame, hud)
    assert out.shape == frame.shape and np.array_equal(frame, before)
    assert not np.array_equal(out, frame)
    assert overlays.format_conditions(None) == "Weather data unavailable"
    assert overlays.format_conditions({"temp": 71.6, "feels_like": 60, "description": "rain", "wind_speed": 20}) \
        == "72 F, feels 60, rain, wind 20 mph"


def test_quote_overlay_fits_very_long_text(frame):
    quote = "Perseverance is not a long race; it is many short races one after the other. " * 8
    out = overlays.add_quote_overlay(frame.copy(), quote, "Walter Elliot", title="A Very Long Thought")
    assert out.shape == frame.shape
    # Nothing may be drawn over the status bar region.
    bar_top = config.FRAME_HEIGHT - overlays.status_bar_height()
    assert np.array_equal(out[bar_top + 4:], frame[bar_top + 4:])


def test_news_and_forecast_overlays(frame):
    story = dict(demo_news()[0], summary=demo_news()[0]["summary"] * 4)
    out = overlays.add_news_overlay(frame.copy(), story, fetched_at=None)
    assert out.shape == frame.shape
    assert overlays.add_news_overlay(frame.copy(), {}).shape == frame.shape

    out = overlays.add_forecast_overlay(frame.copy(), demo_five_day(), demo_weather())
    assert out.shape == frame.shape
    assert overlays.add_forecast_overlay(frame.copy(), None).shape == frame.shape


def test_touch_buttons_hit_test_and_autohide(monkeypatch):
    buttons = build_mode_buttons()
    assert [b["mode"] for b in buttons] == [m["mode"] for m in config.MODE_DEFINITIONS]
    x1, y1, x2, y2 = buttons[1]["rect"]
    assert hit_test(buttons, (x1 + x2) // 2, (y1 + y2) // 2) == buttons[1]["mode"]
    assert hit_test(buttons, 5, 400) is None

    hud = HudState(buttons)
    hud.touch((x1 + x2) // 2, (y1 + y2) // 2)   # first tap only reveals
    assert hud.show_buttons and not hud.mode_dirty
    hud.touch((x1 + x2) // 2, (y1 + y2) // 2)   # second tap selects
    assert hud.mode == buttons[1]["mode"] and hud.mode_dirty

    monkeypatch.setattr(config, "BUTTON_AUTOHIDE_SECONDS", 0)
    hud.tick()
    assert not hud.show_buttons and hud.needs_redraw
