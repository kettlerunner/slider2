# Slider2

Slider2 is a Python slideshow for a Raspberry Pi with a touchscreen. It shows photos and videos from a Google Drive folder with transition effects, and overlays the time, current conditions, a 5-day forecast, an AI weather briefing, motivational quotes, and an AI news digest.

## Features

- Syncs images and videos from a Google Drive folder in the background (paginated listing, trashed files ignored, atomic downloads, removed files cleaned up locally)
- Pre-scales every photo once to display size so the render loop never decodes a 12-megapixel JPEG; EXIF orientation, 16-bit PNGs, PNG transparency, and GIFs are handled
- 9 transitions (fade, slide left/right, wipe top/bottom, melt, wave, ripple, petal bloom), all vectorized in OpenCV and driven by the clock so they last exactly `transition_time` on any hardware
- Status bar with the time, temperature, feels-like, conditions, and wind
- 5-day forecast with drawn weather icons, precipitation chance, and sunrise/sunset
- AI weather blurbs in four styles (practical briefing, poem, haiku, zen master) built from current conditions, the coming hours, and the outlook
- AI news digest with web search: category, neutral summary, "why it matters", sources, and a bias label, returned as structured JSON
- Random quotes from `quotes.json` (every quote plays before any repeats)
- Touchscreen mode buttons (Random, News, Weather, Pics, Video), auto-hidden after a few seconds
- Auto-dims to black when the room is dark (camera-based)
- Never blocks the screen on the network: weather, news, Drive sync, and the camera all run in background threads
- Auto-updates from GitHub on startup via `git pull`
- All settings in `config.json`; API keys in environment variables

## Project structure

```
slider2/
  app.py               # Launcher (git pull + restart-on-crash supervisor)
  config.json          # All user-configurable settings
  quotes.json          # Motivational quotes
  requirements.txt     # Python dependencies
  slider/
    config.py          # Config loader (config.json + env vars)
    slideshow.py       # Main loop: slide selection, holds, transitions, touch
    services.py        # Background threads + SharedState (media, data, sensor)
    playlist.py        # Shuffle-bag playlist with mode filtering
    display.py         # OpenCV window, HUD state, presentation
    overlays.py        # Status bar, forecast, news, and quote panels
    transitions.py     # Transition effects
    image_processing.py# Loading, pre-scaled copies, backgrounds, collages
    video.py           # Real-time video playback
    weather.py         # OpenWeatherMap client, caching, icons
    forecast_summary.py# AI weather blurbs
    news.py            # AI news digest
    ai_client.py       # OpenAI wrapper (Responses API, structured output)
    drive_sync.py      # Google Drive sync
    brightness.py      # Ambient light sensor (camera)
    touch_ui.py        # Mode buttons
    quotes.py, cursor.py, utils.py, demo.py
  tests/               # pytest suite (no network or display needed)
```

## Requirements

Python 3.9 or later.

```bash
pip install -r requirements.txt
```

On Raspberry Pi OS, OpenCV is usually installed via apt:

```bash
sudo apt install python3-opencv
pip install -r requirements.txt
```

You also need:

- A Google Drive `credentials.json` (OAuth desktop client) in the project directory
- An OpenWeatherMap API key
- An OpenAI API key (for the news digest and weather blurbs)

## Configuration

All settings live in `config.json`. Keys and their defaults:

| Section | Keys |
| --- | --- |
| `display` | `width` 800, `height` 480, `transition_time` 2, `display_time` 30, `fps` 30, `windowed` false |
| `timezone` | IANA zone for the clock and schedule, default `America/Chicago` |
| `display_mix` | `day_start_hour`, `day_end_hour`, and per-layout weights for `weekday`, `weekend`, `evening` |
| `weather` | `current_city`, `forecast_city`, `country_code` |
| `news` | `topics` (list), `local_area` (one local story per digest, empty to disable), `max_age_hours` 48 |
| `google_drive` | `folder_id` |
| `camera` | `enabled`, `index`, `brightness_dark_threshold`, `brightness_check_interval_seconds`, `retry_interval_seconds` |
| `openai` | `chat_model`, `news_model` (default `gpt-5-mini`), `reasoning_effort` (`low`), `request_timeout` |
| `cache_ttl` | `media_refresh_minutes` 2, `weather_minutes` 15, `forecast_minutes` 15, `ai_success_minutes` 30, `ai_failure_minutes` 5, `news_pool_refresh_hours` 2, `news_failure_minutes` 3, `news_pool_size` 6 |
| `modes` | Touch button definitions (`mode`, `label`) |
| `low_power_mode` | `auto` (ARM Linux detection), `true`, or `false` |

Environment variables:

- `OPENAI_API_KEY` (required for AI features), `OPENAI_CHAT_MODEL`, `OPENAI_NEWS_MODEL`, `OPENAI_REASONING_EFFORT`, `OPENAI_API_BASE`
- `WEATHERMAP_API_KEY` (or `OPENWEATHERMAP_API_KEY`)
- `SLIDER_TIMEZONE`, `SLIDER_DRIVE_FOLDER_ID`, `SLIDER_FORCE_LOW_POWER`

Development helpers (environment only):

- `SLIDER_WINDOWED=1` runs in a normal window instead of fullscreen
- `SLIDER_DEMO=1` uses built-in sample weather and news instead of the network
- `SLIDER_DISPLAY_TIME` and `SLIDER_TRANSITION_TIME` override the timings in seconds
- `SLIDER_CAPTURE_DIR=path` saves a PNG of the screen every `SLIDER_CAPTURE_SECONDS` (default 2)

## Running

1. Put `credentials.json` in the project directory.
2. Export `OPENAI_API_KEY` and `WEATHERMAP_API_KEY`.
3. Run:

```bash
python app.py
```

On first run a browser opens to authorize Google Drive access. `app.py` pulls the latest code from GitHub, then launches the slideshow and restarts it if it ever crashes. You can also run the slideshow directly:

```bash
python -m slider
```

Press `q` or `Esc` to quit. Tap the screen once to show the mode buttons, then tap a mode.

To try it on a desktop without any keys or Drive access, drop a few photos in `images/` and run:

```bash
SLIDER_WINDOWED=1 SLIDER_DEMO=1 SLIDER_DISPLAY_TIME=5 python -m slider
```

## Tests

```bash
python -m pytest tests/
```

## Quotes

`quotes.json` holds the motivational quotes. Add your own in the same format.

## License

Slider2 is distributed under the terms of the GNU General Public License version 3. See the [LICENSE](LICENSE) file for details.
