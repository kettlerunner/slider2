# Slider2

Slider2 is a Python-based slideshow application designed for a Raspberry Pi with a touchscreen. It displays images and videos from a Google Drive folder with transition effects, overlaying the current time, weather data, motivational quotes, AI-generated weather summaries, 5-day forecasts, and current news.

## Features

- Downloads images and videos from a specified Google Drive folder
- 9 transition effects (fade, slide, wipe, melt, wave, ripple, petal bloom)
- Displays current time and local weather in a status bar
- 5-day forecast overlay using OpenWeatherMap
- AI-generated weather summaries (poem, haiku, zen master styles) via OpenAI
- AI-generated news summaries with source diversity and bias labels via OpenAI web search
- Random motivational quotes from `quotes.json`
- Touchscreen mode buttons (Random, News, Weather, Pics, Video)
- Auto-dims when room lights are off (camera-based ambient brightness)
- Auto-updates from GitHub on startup via `git pull`
- All settings configurable via `config.json` (no code editing required)

## Project Structure

```
slider2/
  app.py              # Launcher (git pull + start slideshow)
  config.json         # All user-configurable settings
  quotes.json         # Motivational quotes
  requirements.txt    # Python dependencies
  slider/             # Main application package
    config.py         # Config loader
    ai_client.py      # OpenAI API (Responses + Chat Completions)
    drive_sync.py     # Google Drive sync
    weather.py        # Weather data + icons
    news.py           # AI news generation
    forecast_summary.py  # AI weather summaries
    image_processing.py  # Image resize, blur, stitch
    transitions.py    # Slide transition effects
    overlays.py       # On-screen overlays (forecast, news, time, quotes)
    display.py        # Window management + frame rendering
    slideshow.py      # Main loop + orchestrator
    ...               # cursor, brightness, touch_ui, video, quotes, utils
```

## Requirements

Python 3.9 or later. Install dependencies:

```bash
pip install -r requirements.txt
```

On Raspberry Pi OS, OpenCV is typically installed via apt:

```bash
sudo apt install python3-opencv
pip install -r requirements.txt
```

You also need:
- A Google Drive `credentials.json` file for OAuth
- An OpenWeatherMap API key
- An OpenAI API key (for news and weather summaries)

## Configuration

All settings live in `config.json`. Edit this file instead of source code:

```json
{
  "display": { "width": 800, "height": 480, "display_time": 30 },
  "weather": { "current_city": "Waupun", "forecast_city": "Fond du Lac" },
  "google_drive": { "folder_id": "your-folder-id-here" },
  "camera": { "index": 1, "brightness_dark_threshold": 35.0 },
  "openai": { "chat_model": "gpt-4o-mini", "news_model": "gpt-4o-mini" },
  "low_power_mode": "auto"
}
```

If `config.json` is missing, all values use sensible defaults.

Environment variables override config values:
- `OPENAI_API_KEY` - OpenAI API key (required for AI features)
- `OPENAI_CHAT_MODEL` - Model for weather summaries (default: `gpt-4o-mini`)
- `OPENAI_NEWS_MODEL` - Model for news summaries (default: `gpt-4o-mini`)
- `WEATHERMAP_API_KEY` - OpenWeatherMap API key (required for weather)

## Running the Slideshow

1. Place your `credentials.json` in the project directory.
2. Set environment variables (`OPENAI_API_KEY`, `WEATHERMAP_API_KEY`).
3. Run:

```bash
python app.py
```

On first run, a browser will open to authorize Google Drive access.

`app.py` pulls the latest code from GitHub, then launches the slideshow. You can also run the slideshow directly:

```bash
python -m slider
```

## Quotes

The `quotes.json` file contains motivational quotes displayed randomly during the slideshow. Add your own in the same format.

## License

Slider2 is distributed under the terms of the GNU General Public License version 3. See the [LICENSE](LICENSE) file for details.
