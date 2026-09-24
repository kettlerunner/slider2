# Agent Instructions

This project runs on a Raspberry Pi 4 using the current Raspberry Pi OS and the official 7-inch Raspberry Pi touchscreen (800x480). Prioritize solutions that remain lightweight and robust on this hardware. Ensure the slideshow keeps running continuously even when network connections are unreliable.

## Architecture

The application is a Python package (`slider/`) launched via `app.py` or `python -m slider`.

- `config.py` — All settings from `config.json` + env vars. No hardcoded values elsewhere.
- `slideshow.py` — Main loop (main thread only). Reads a `SharedState` snapshot, builds frames, runs transitions, handles touch. It must never block on network or disk beyond loading a pre-scaled image.
- `services.py` — Background threads: `media` (Drive sync + pre-scaled display copies), `data` (weather, forecasts, AI summary, news), `sensor` (camera brightness). They publish into `SharedState`; the main loop only reads snapshots.
- `playlist.py` — Shuffle-bag ordering, mode filtering, and refresh-merging (a Drive sync never restarts the show).
- `transitions.py` — Factories returning `render(alpha)`; driven by the wall clock. All per-pixel work is OpenCV/NumPy; no Python loops over pixels and no float conversions of whole frames per frame.
- `overlays.py` / `display.py` — Frames are always BGR uint8 at frame size. Overlays draw in place; `draw_hud` copies. Blend only the region you touch.
- `ai_client.py` — The only module that touches the OpenAI SDK (Responses API with structured output; degrades gracefully).
- `weather.py`, `news.py`, `forecast_summary.py` — Network code with disk caches that always fall back to the last good value.

## Rules

- Keep all configuration in `config.json` / `config.py`. Never hardcode cities, API keys, folder IDs, timezones, or display dimensions in other modules.
- OpenAI calls go through `ai_client.py` — never call the SDK directly from other modules.
- All network operations must fail gracefully (return None / cached values). The slideshow must never crash or stall because of network issues; anything slow belongs in `services.py`.
- Only the main thread may call OpenCV GUI functions (`imshow`, `waitKey`, window properties).
- Test with `python -m pytest tests/` from the project root. Tests must not need a display, camera, or network.
- To try changes on a desktop: `SLIDER_WINDOWED=1 SLIDER_DEMO=1 SLIDER_DISPLAY_TIME=5 python -m slider` with a few images in `images/`.
