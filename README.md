# Slider2

Slider2 is a Python-based slideshow application that displays images and videos from a Google Drive folder. Each slide can include overlays showing the current time, weather data, short quotes, or a five‑day forecast. Transitions and overlays are generated with OpenCV and other Python libraries.

## Features

- Downloads images and videos from a specified Google Drive folder
- Supports multiple transition effects between slides
- Displays the current time and local weather
- Optional overlays for daily or 5‑day forecasts using OpenWeatherMap
- Can show random motivational quotes from `quotes.json`
- Plays local video files in addition to images
- Fetches data from OpenAI to generate short weather summaries
- Uses OpenAI web search to retrieve and summarize current news with source diversity and bias labels

## Requirements

The script uses Python 3.11 or later with the following packages:

- `opencv-python`
- `numpy`
- `requests`
- `google-auth`, `google-auth-oauthlib`, `google-api-python-client`
- `openai`
- `svglib`, `reportlab`

Install them with pip:

```bash
pip install opencv-python numpy requests google-auth google-auth-oauthlib \
    google-api-python-client openai svglib reportlab
```

You also need a Google Drive `credentials.json` file for OAuth and a valid
OpenWeatherMap API key.

## Running the Slideshow

1. Place your `credentials.json` in the project directory.
2. Set the following environment variables:
   - `OPENAI_API_KEY` – OpenAI API key for news and forecast summaries.
   - `OPENAI_CHAT_MODEL` – (Optional) OpenAI model for forecast summaries (default: `gpt-5.2-mini`).
   - `OPENAI_NEWS_MODEL` – (Optional) OpenAI model for news summaries (default: `gpt-5.2-mini`, set independently).
   - `WEATHERMAP_API_KEY` – OpenWeatherMap API key for weather data.
3. Edit `slider.py` to set your Google Drive `folder_id`.
4. Run the slideshow:

```bash
python app.py
```

The first run will open a browser to authorize Google Drive access and create a
`token.json` file. After updating any files from GitHub, `app.py` launches
`slider.py` which handles the slideshow and transitions.

## Quotes

The `quotes.json` file contains motivational quotes displayed randomly during
the slideshow. You can edit this file or add your own quotes in the same
format.

## License

Slider2 is distributed under the terms of the GNU General Public License
version 3. See the [LICENSE](LICENSE) file for details.
