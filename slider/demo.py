"""Canned weather and news data for SLIDER_DEMO=1 (no network or API keys needed)."""

from datetime import datetime, timedelta

from slider import config
from slider.utils import now_local

_DAY_PATTERNS = [
    ("scattered clouds", 61, 78, 0.05),
    ("light rain", 58, 70, 0.65),
    ("clear sky", 55, 74, 0.0),
    ("overcast clouds", 60, 72, 0.2),
    ("snow", 28, 36, 0.8),
]


def demo_weather():
    now = now_local()
    sunrise = now.replace(hour=6, minute=45, second=0, microsecond=0)
    sunset = now.replace(hour=18, minute=55, second=0, microsecond=0)
    return {
        "temp": 71.6, "feels_like": 70.2, "temp_min": 64.0, "temp_max": 78.0,
        "humidity": 55.0, "wind_speed": 9.0, "wind_gust": 14.0,
        "main": "Clouds", "description": "scattered clouds",
        "sunrise": sunrise.timestamp(), "sunset": sunset.timestamp(),
        "city": config.WEATHER_CURRENT_CITY,
    }


def demo_today():
    now = now_local()
    start = now.replace(minute=0, second=0, microsecond=0)
    slices = []
    for i in range(5):
        stamp = start + timedelta(hours=3 * i)
        desc, lo, hi, pop = _DAY_PATTERNS[i % 2]
        slices.append({
            "time": stamp.strftime("%I:%M %p").lstrip("0"),
            "temp": lo + (hi - lo) * (0.5 + 0.1 * i), "feels_like": lo + 4,
            "description": desc, "wind_speed": 6 + i, "humidity": 50 + 3 * i, "pop": pop,
        })
    return slices


def demo_five_day():
    today = now_local().date()
    days = []
    for i in range(5):
        day = today + timedelta(days=i)
        desc, lo, hi, pop = _DAY_PATTERNS[i]
        days.append({
            "date": day.strftime("%A, %b %d"), "date_key": day.isoformat(),
            "temp_min": float(lo), "temp_max": float(hi), "description": desc, "pop": pop,
        })
    return days


def demo_summary():
    return (
        "Mild and mostly cloudy this afternoon with a light breeze, then showers move in after dark.\n"
        "Grab a light jacket for the evening and keep an umbrella by the door.",
        "briefing",
    )


def demo_news():
    return [
        {
            "headline": "Fed Holds Rates Steady, Signals Cuts Later This Year",
            "summary": "The Federal Reserve left its benchmark rate unchanged at its September meeting. "
                       "Officials said inflation is easing but want more data before lowering borrowing costs.",
            "why_it_matters": "Mortgage, car loan, and credit card rates are likely to stay where they are for now.",
            "category": "Economy", "published_at": "",
            "sources": ["Reuters", "AP News", "Wall Street Journal"],
            "bias": "Center", "bias_note": "mixed outlets, factual reporting",
        },
        {
            "headline": "Chipmakers Announce New Wisconsin Packaging Plant",
            "summary": "A consortium of semiconductor firms will build an advanced packaging facility near Madison, "
                       "creating an estimated 1,200 jobs. Construction is expected to begin next spring.",
            "why_it_matters": "New manufacturing jobs and suppliers could lift the regional economy.",
            "category": "Local", "published_at": "",
            "sources": ["Milwaukee Journal Sentinel", "Wisconsin Public Radio"],
            "bias": "Center", "bias_note": "local business coverage",
        },
        {
            "headline": "Trade Talks Resume as Tariff Deadline Approaches",
            "summary": "Negotiators from the US and EU met in Brussels to avert new tariffs on autos and steel. "
                       "Both sides described the talks as constructive but no agreement was announced.",
            "why_it_matters": "Tariffs would raise prices on imported cars and appliances.",
            "category": "World", "published_at": "",
            "sources": ["BBC", "Financial Times", "Politico"],
            "bias": "Center", "bias_note": "international outlets, balanced framing",
        },
    ]


def apply_demo_data(state):
    """Publish the canned data set into a SharedState."""
    state.update(
        weather=demo_weather(),
        today_forecast=demo_today(),
        forecast_5day=demo_five_day(),
        forecast_summary=demo_summary(),
        news_pool=demo_news(),
        news_fetched_at=datetime.now(),
    )
