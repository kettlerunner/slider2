from __future__ import print_function
import os
import io
import json
import random
import cv2
import numpy as np
from datetime import datetime, timedelta
import requests
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from openai import OpenAI
import textwrap
import re
from collections import defaultdict
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def sanitize_text(text):
    # Replace common problematic characters
    replacements = {
        '"': '"',
        '"': '"',
        ''': "'",
        ''': "'",
        '–': '-',
        '—': '-',
        '…': '...'
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove any remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    
    return text

# Constants
frame_width = 800
frame_height = 480
transition_time = 2
display_time = 10
num_transition_frames = int(transition_time * 30)
api_key = os.getenv('WEATHERMAP_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_key)

# If modifying these SCOPES, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_weather_forecast(api_key, city="Waupun", country_code="US"):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},{country_code}&units=imperial&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        daily_forecast = defaultdict(lambda: {'temp_min': float('inf'), 'temp_max': float('-inf'), 'descriptions': []})
        
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'])
            date_key = dt.strftime("%Y-%m-%d")
            
            # Update min and max temperatures
            daily_forecast[date_key]['temp_min'] = min(daily_forecast[date_key]['temp_min'], item['main']['temp_min'])
            daily_forecast[date_key]['temp_max'] = max(daily_forecast[date_key]['temp_max'], item['main']['temp_max'])
            
            # Store weather description
            daily_forecast[date_key]['descriptions'].append(item['weather'][0]['description'])
            
            # If this is the noon forecast, use it as the representative weather
            if dt.hour == 12:
                daily_forecast[date_key]['main_description'] = item['weather'][0]['description']
        
        # Format the result
        formatted_forecast = []
        for date, forecast in daily_forecast.items():
            dt = datetime.strptime(date, "%Y-%m-%d")
            formatted_forecast.append({
                'date': dt.strftime("%A, %b %d"),
                'temp_min': forecast['temp_min'],
                'temp_max': forecast['temp_max'],
                'description': forecast.get('main_description', max(set(forecast['descriptions']), key=forecast['descriptions'].count))
            })
        
        # Sort by date and return only the next 5 days
        formatted_forecast.sort(key=lambda x: datetime.strptime(x['date'], "%A, %b %d"))
        return formatted_forecast[:5]
    else:
        print("Error fetching weather forecast data")
        return None

icon_images = {
    'clear': cv2.imread('icons/sunny.png', cv2.IMREAD_UNCHANGED),
    'cloudy': cv2.imread('icons/cloudy.png', cv2.IMREAD_UNCHANGED),
    'rain': cv2.imread('icons/rain.png', cv2.IMREAD_UNCHANGED),
    'snow': cv2.imread('icons/snow.png', cv2.IMREAD_UNCHANGED),
    'windy': cv2.imread('icons/windy.png', cv2.IMREAD_UNCHANGED)
}

def get_weather_icon(description):
    """Return the appropriate icon image based on the weather description."""
    description = description.lower()
    if 'clear' in description:
        return icon_images.get('clear')
    elif 'cloud' in description:
        return icon_images.get('cloudy')
    elif 'rain' in description:
        return icon_images.get('rain')
    elif 'windy' in description:
        return icon_images.get('windy')

def get_ai_generated_news():
    prompt = """
            Search for current technology, international, business, or economic news relevant to the last week or even today specifically. Dig sources available to you and respond with a compelling title and summary of the news that you are reporting.

            Focus only on current events. Make sure you are pulling events from the past week.

            Respond in JSON format:

            {
              "headline": "The headline goes here",
              "summary": "A brief 1-2 sentence summary of the news"
            }
        """

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        
        news_data = completion.choices[0].message.content.strip()
        if news_data.startswith('```json'):
            news_data = news_data[len('```json'):].strip()
        if news_data.endswith('```'):
            news_data = news_data[:-len('```')].strip()
        
        news_json = json.loads(news_data)
        return news_json
    except Exception as e:
        print(f"Error generating news: {e}")
        return {"headline": "Error generating news", "summary": "Please try again later."}

def add_forecast_overlay(frame, forecast):
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        thickness = 1

        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 200), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Title
        cv2.putText(frame, "5-Day Forecast", (10, 30), font, 0.8, font_color, 2, cv2.LINE_AA)

        if forecast:
            col_width = frame.shape[1] // 5
            for i, day in enumerate(forecast):
                x = i * col_width + 10
                y = 60
                
                # Day and date
                cv2.putText(frame, day['date'].split(',')[0], (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
                y += 25
                cv2.putText(frame, day['date'].split(',')[1].strip(), (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
                y += 35

                # Temperature range
                temp_text = f"{day['temp_min']:.1f} - {day['temp_max']:.1f} F"
                cv2.putText(frame, temp_text, (x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
                y += 35

                # Weather icon overlay
                icon_img = get_weather_icon(day['description'])
                if icon_img is not None:
                    icon_size = 32  # Icon size, adjust as needed
                    icon_img = cv2.resize(icon_img, (icon_size, icon_size))
                    icon_y_offset = y - 20  # Position adjustment
                    icon_x_offset = x + 10
                    y += 40

                    # Check if icon has transparency (alpha channel)
                    if icon_img.shape[2] == 4:
                        for c in range(3):  # Apply BGR channels
                            frame[icon_y_offset:icon_y_offset+icon_size, icon_x_offset:icon_x_offset+icon_size, c] = \
                                icon_img[:, :, c] * (icon_img[:, :, 3] / 255.0) + frame[icon_y_offset:icon_y_offset+icon_size, icon_x_offset:icon_x_offset+icon_size, c] * (1 - icon_img[:, :, 3] / 255.0)
                    else:
                        frame[icon_y_offset:icon_y_offset+icon_size, icon_x_offset:icon_x_offset+icon_size] = icon_img

                # Description
                desc = day['description']
                cv2.putText(frame, desc, (x, y), font, font_scale * 0.8, font_color, thickness, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Forecast data unavailable", (10, 100), font, font_scale, font_color, thickness, cv2.LINE_AA)
        
        return frame
    except Exception as e:
        print(f"Error adding forecast overlay: {e}")
        return frame

def add_news_overlay(frame, news):
    try:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 1
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - 240, frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        headline = textwrap.wrap(news['headline'], width=60)
        summary = textwrap.wrap(news['summary'], width=60)
        y = frame.shape[0] - 220
        
        cv2.putText(frame, "News Update:", (10, y), font, font_scale * 1, font_color, thickness, cv2.LINE_AA)
        y += 30
        
        for line in headline:
            cv2.putText(frame, line, (10, y), font, font_scale, font_color, thickness, cv2.LINE_AA)
            y += 30
        
        for line in summary:
            cv2.putText(frame, line, (10, y), font, font_scale * 0.8, font_color, thickness, cv2.LINE_AA)
            y += 25
        
        return frame
    except Exception as e:
        print(f"Error adding news overlay: {e}")
        return frame

def get_weather_data(api_key, cache_file='weather_cache.json'):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            data = json.load(f)
            timestamp = datetime.strptime(data['timestamp'], "%Y-%m-%d %H:%M:%S")
            if datetime.now() - timestamp < timedelta(minutes=15):
                return data['temp'], data['weather']
    url = f"http://api.openweathermap.org/data/2.5/weather?q=Waupun,WI,US&units=imperial&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        temp = data['main']['temp']
        weather = data['weather'][0]['main']
        with open(cache_file, 'w') as f:
            json.dump({'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 'temp': temp, 'weather': weather}, f)
        return temp, weather
    else:
        print("Error fetching weather data")
        return None, None

def authenticate_drive():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name, modifiedTime, size)").execute()
    items = results.get('files', [])
    return items

def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()

def resize_and_pad(image, width, height):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    has_alpha = image.shape[2] == 4
    if has_alpha:
        padded_image = np.zeros((height, width, 4), dtype=np.uint8)
        padded_image[:,:,3] = 0
    else:
        padded_image = np.zeros((height, width, 3), dtype=np.uint8)

    top_pad = (height - resized_image.shape[0]) // 2
    left_pad = (width - resized_image.shape[1]) // 2

    if has_alpha:
        padded_image[top_pad:top_pad+resized_image.shape[0], 
                     left_pad:left_pad+resized_image.shape[1], :] = resized_image
    else:
        padded_image[top_pad:top_pad+resized_image.shape[0], 
                     left_pad:left_pad+resized_image.shape[1], :] = resized_image
        
    return padded_image

def create_zoomed_blurred_background(image, width, height):
    image_copy = image.copy()
    h, w = image_copy.shape[:2]
    scale = max(width / w, height / h) * 1.1  # Zoom in a bit more
    zoomed_image = cv2.resize(image_copy, (int(w * scale), int(h * scale)))

    start_x = (zoomed_image.shape[1] - width) // 2
    start_y = (zoomed_image.shape[0] - height) // 2
    cropped_image = zoomed_image[start_y:start_y + height, start_x:start_x + width]

    blurred_image = cv2.GaussianBlur(cropped_image, (51, 51), 0)
    return blurred_image

def create_blurred_background(image, width, height):
    h, w = image.shape[:2]
    scale = max(width / w, height / h)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))

    start_x = (resized_image.shape[1] - width) // 2
    start_y = (resized_image.shape[0] - height) // 2
    centered_image = resized_image[start_y:start_y + height, start_x:start_x + width]

    blurred_image = cv2.GaussianBlur(centered_image, (51, 51), 0)
    return blurred_image

def ensure_same_channels(img1, img2):
    if img1.shape[2] != img2.shape[2]:
        if img1.shape[2] == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2BGRA)
        else:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)
    return img1, img2

def fade_transition(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for alpha in np.linspace(0, 1, num_frames):
        blended_frame = cv2.addWeighted(current_img, 1 - alpha, next_img, alpha, 0)
        yield blended_frame

def slide_transition_left(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for i in range(num_frames):
        alpha = i / num_frames
        dx = int(current_img.shape[1] * alpha)
        frame = np.zeros_like(current_img)
        frame[:, :current_img.shape[1]-dx] = current_img[:, dx:]
        frame[:, current_img.shape[1]-dx:] = next_img[:, :dx]
        yield frame

def slide_transition_right(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for i in range(num_frames):
        alpha = i / num_frames
        dx = int(current_img.shape[1] * alpha)
        frame = np.zeros_like(current_img)
        frame[:, dx:] = current_img[:, :current_img.shape[1]-dx]
        frame[:, :dx] = next_img[:, current_img.shape[1]-dx:]
        yield frame

def wipe_transition_top(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for i in range(num_frames):
        alpha = i / num_frames
        dy = int(current_img.shape[0] * alpha)
        frame = np.zeros_like(current_img)
        frame[:current_img.shape[0]-dy, :] = current_img[dy:, :]
        frame[current_img.shape[0]-dy:, :] = next_img[:dy, :]
        yield frame

def wipe_transition_bottom(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for i in range(num_frames):
        alpha = i / num_frames
        dy = int(current_img.shape[0] * alpha)
        frame = np.zeros_like(current_img)
        frame[dy:, :] = current_img[:current_img.shape[0]-dy, :]
        frame[:dy, :] = next_img[current_img.shape[0]-dy:, :]
        yield frame

def stitch_images(images, width, height):
    num_images = len(images)
    
    stitched_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    background_image = random.choice(images)
    if (background_image.shape[2] == 4):
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGRA2BGR)
    blurred_background = create_blurred_background(background_image, width, height)
    
    cv2.addWeighted(stitched_image, 0.5, blurred_background, 0.5, 0, stitched_image)

    rows = cols = int(np.ceil(np.sqrt(num_images)))
    grid_width = width // cols
    grid_height = height // rows

    for i, image in enumerate(images):
        resized_image = resize_and_pad(image, grid_width, grid_height)
        row = i // cols
        col = i % cols
        start_x = col * grid_width
        start_y = row * grid_height
        
        if resized_image.shape[2] == 4:
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
            alpha = resized_image[:,:,3] / 255.0
            alpha = np.repeat(alpha[:,:,np.newaxis], 3, axis=2)
            stitched_image[start_y:start_y+grid_height, start_x:start_x+grid_width] = \
                (1-alpha) * stitched_image[start_y:start_y+grid_height, start_x:start_x+grid_width] + \
                alpha * rgb_image
        else:
            stitched_image[start_y:start_y+grid_height, start_x:start_x+grid_width] = resized_image

    return stitched_image

def create_single_image_with_background(image, width, height):
    if image.shape[2] == 4:
        bg_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bg_image = image.copy()
    
    blurred_background = create_zoomed_blurred_background(bg_image, width, height)
    resized_image = resize_and_pad(image, width, height)
    
    top_pad = (height - resized_image.shape[0]) // 2
    left_pad = (width - resized_image.shape[1]) // 2

    if resized_image.shape[2] == 4:
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
        alpha = resized_image[:,:,3] / 255.0
        alpha = np.repeat(alpha[:,:,np.newaxis], 3, axis=2)
        background = blurred_background[top_pad:top_pad+resized_image.shape[0], left_pad:left_pad+resized_image.shape[1]]
        combined = (1-alpha) * background + alpha * rgb_image
        blurred_background[top_pad:top_pad+resized_image.shape[0], left_pad:left_pad+resized_image.shape[1]] = combined
    else:
        blurred_background[top_pad:top_pad+resized_image.shape[0], left_pad:left_pad+resized_image.shape[1]] = resized_image

    return blurred_background

def add_time_overlay(frame, temp, weather):
    try:
        time_text = datetime.now().strftime("%B %d %Y, %I:%M %p")
        weather_text = f"Temp: {temp:.1f} F" if temp is not None else "Weather data unavailable"
        if weather:
            if "rain" in weather.lower():
                if "drizzle" in weather.lower():
                    weather_text += ", Raining"
                else:
                    weather_text += ", Rain Predicted"
            elif "snow" in weather.lower():
                if "flurries" in weather.lower():
                    weather_text += ", Snowing"
                else:
                    weather_text += ", Snow Predicted"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_color = (255, 255, 255)
        thickness = 1

        text_size_time, _ = cv2.getTextSize(time_text, font, font_scale, thickness)
        text_size_weather, _ = cv2.getTextSize(weather_text, font, font_scale, thickness)

        text_x_weather = 10
        text_y = frame.shape[0] - 12
        text_x_time = frame.shape[1] - text_size_time[0] - 10

        overlay_frame = frame.copy()
        bar_height = max(text_size_time[1], text_size_weather[1]) + 20
        overlay = overlay_frame.copy()
        cv2.rectangle(overlay, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        alpha = 0.8
        cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)

        cv2.putText(overlay_frame, weather_text, (text_x_weather, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.putText(overlay_frame, time_text, (text_x_time, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

        return overlay_frame
    except Exception as e:
        print(f"Error adding overlay: {e}")
        return frame

import json
import random

def get_random_quote():
    prompts = [
        """Generate an uncommon, thought-provoking quote. Draw from a wide range of sources including:
        - Ancient wisdom traditions
        - Modern thinkers and innovators
        - Indigenous cultures
        - Scientists and researchers
        - Artists and creatives
        - Activists and change-makers
        - Philosophers from various schools of thought
        - Literary figures from different genres

        Prioritize lesser-known quotes that offer unique perspectives or challenge conventional wisdom. Avoid commonly cited or viral quotes. Respond only with the quote and source in this JSON format:

        {
          "quote": "The quote text goes here.",
          "source": "Name, brief description of who they are"
        }""",

        """Create an inspiring quote that feels fresh and original. Consider these approaches:
        - Combine ideas from different fields (e.g., science and art, technology and nature)
        - Use unexpected metaphors or analogies
        - Offer a counterintuitive perspective
        - Frame a common idea in a new way
        - Focus on emerging global challenges or opportunities

        The quote should be concise but impactful. Attribute it to a real but not widely known figure, or create a plausible fictional source. Respond only in this JSON format:

        {
          "quote": "The quote text goes here.",
          "source": "Name, brief description of who they are"
        }""",

        """Generate an inspirational quote based on these random elements:

        1. Choose one: [Nature, Technology, Human Relationships, Personal Growth, Social Change]
        2. Emotion to evoke: [Wonder, Determination, Empathy, Curiosity, Courage]
        3. Quote length: [Under 10 words, 10-20 words, 20-30 words]
        4. Source era: [Ancient (pre-1500), Early Modern (1500-1800), Modern (1800-1950), Contemporary (1950-present)]
        5. Cultural region: [Africa, Asia, Europe, North America, South America, Oceania, Middle East]

        Create a quote that incorporates these elements in an unexpected way. The source should be a real but not famous person from the chosen era and region. Respond only in this JSON format:

        {
          "quote": "The quote text goes here.",
          "source": "Name, brief description including their era and cultural background"
        }"""
    ]

    chosen_prompt = random.choice(prompts)

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": chosen_prompt}
        ]
    )

    print(completion.choices[0].message.content)
    try:
        # Extract the response content and clean it up
        quote_data = completion.choices[0].message.content.strip()
        
        # Ensure the response is a valid JSON string
        if quote_data.startswith('```json'):
            quote_data = quote_data[len('```json'):].strip()
        if quote_data.endswith('```'):
            quote_data = quote_data[:-len('```')].strip()
        quote_json = json.loads(quote_data)
        return quote_json.get("quote", ""), quote_json.get("source", "")
    except Exception as e:
        print(f"Error getting quote: {e}")
        return "", ""

def add_quote_overlay(frame, quote, source):
    try:
        # Sanitize the text
        quote = sanitize_text(quote)
        source = sanitize_text(source)
        
        # Define fonts and scales
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_quote = 0.8
        font_scale_source = 0.6
        font_color = (105, 105, 105)  # Dark gray color
        thickness = 1
        
        # Maximum width for text
        max_width = frame.shape[1] - 40  # Padding of 20 on each side
        
        # Split the quote and source into multiple lines
        quote_lines = textwrap.wrap(quote, width=50)
        source_lines = textwrap.wrap(f"- {source}", width=50)
        
        # Calculate text size and total height
        text_height = 0
        line_height_quote = cv2.getTextSize("Test", font, font_scale_quote, thickness)[0][1]
        line_height_source = cv2.getTextSize("Test", font, font_scale_source, thickness)[0][1]
        
        text_height += line_height_quote * len(quote_lines)
        text_height += line_height_source * len(source_lines)
        text_height += 20  # Additional padding between quote and source

        # Determine the width of the box
        max_line_width = max(
            [cv2.getTextSize(line, font, font_scale_quote, thickness)[0][0] for line in quote_lines] +
            [cv2.getTextSize(line, font, font_scale_source, thickness)[0][0] for line in source_lines]
        )

        # Box dimensions
        box_width = max_line_width + 40  # Adding padding
        box_height = text_height + 80  # Adding padding
        box_x = (frame.shape[1] - box_width) // 2
        box_y = (frame.shape[0] - box_height) // 2

        # Draw the semi-transparent box
        overlay_frame = frame.copy()
        cv2.rectangle(overlay_frame, (box_x, box_y), (box_x + box_width, box_y + box_height), (255, 255, 255), -1)
        alpha = 0.8  # Transparency factor
        cv2.addWeighted(overlay_frame, alpha, frame, 1 - alpha, 0, overlay_frame)

        # Draw each line of the quote
        y = box_y + 20  # Padding inside the box
        for line in quote_lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale_quote, thickness)
            x = (frame.shape[1] - text_size[0]) // 2  # Center the text horizontally
            cv2.putText(overlay_frame, line, (x, y + text_size[1]), font, font_scale_quote, font_color, thickness, cv2.LINE_AA)
            y += text_size[1] + 10  # Line height + padding

        # Draw each line of the source
        y += 10  # Additional padding before the source
        for line in source_lines:
            text_size, _ = cv2.getTextSize(line, font, font_scale_source, thickness)
            x = (frame.shape[1] - text_size[0]) // 2  # Center the text horizontally
            cv2.putText(overlay_frame, line, (x, y + text_size[1]), font, font_scale_source, font_color, thickness, cv2.LINE_AA)
            y += text_size[1] + 10  # Line height + padding

        return overlay_frame
    except Exception as e:
        print(f"Error adding quote overlay: {e}")
        return frame
        
def load_local_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def save_local_metadata(metadata_file, metadata):
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

def main():
    service = authenticate_drive()

    folder_id = '1hpBzZ_kiXpIBtRv1FN3da8zOhT5J0Ggi'  # Replace with your folder ID
    files = list_files_in_folder(service, folder_id)

    temp_dir = 'images'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    metadata_file = 'metadata.json'
    local_metadata = load_local_metadata(metadata_file)

    images = []
    for file in files:
        file_name = file['name']
        file_path = os.path.join(temp_dir, file_name)
        file_metadata = {
            'name': file_name,
            'modifiedTime': file['modifiedTime'],
            'size': file.get('size', 0)
        }

        if file_name in local_metadata:
            local_file_metadata = local_metadata[file_name]
            if (local_file_metadata['modifiedTime'] == file_metadata['modifiedTime'] and
                local_file_metadata['size'] == file_metadata['size']):
                print(f"Skipping download of unchanged file: {file_name}")
                img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                    images.append(img)
                continue

        print(f"Downloading file: {file_name}")
        download_file(service, file['id'], file_path)
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            images.append(img)

        local_metadata[file_name] = file_metadata

    save_local_metadata(metadata_file, local_metadata)

    if not images:
        print("No images found in the folder.")
        exit()

    random.shuffle(images)

    # Modify the window creation and properties
    cv2.namedWindow('slideshow', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('slideshow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    transitions = [
        fade_transition,
        slide_transition_left,
        slide_transition_right,
        wipe_transition_top,
        wipe_transition_bottom
    ]

    index = 0
    current_img = resize_and_pad(images[index], frame_width, frame_height)
    forecast = get_weather_forecast(api_key)
    #news = get_ai_generated_news()
    while True:
        
        temp, weather = get_weather_data(api_key)
        
        # Randomly choose display type: single image, stitched images, or quote
        display_type = random.choice(["single", "stitch", "quote", "forecast"])
        
        if display_type == "forecast":
            single_image = images[(index + 1) % len(images)]
            next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
            next_img = add_forecast_overlay(next_img, forecast)
        elif display_type == "news":
            single_image = images[(index + 1) % len(images)]
            next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
            news = get_ai_generated_news()
            next_img = add_news_overlay(next_img, news)
        elif display_type == "stitch":
            stitch_count = random.randint(2, 4)
            stitch_indices = random.sample(range(len(images)), stitch_count)
            stitched_images = [images[i] for i in stitch_indices]
            next_img = stitch_images(stitched_images, frame_width, frame_height)
            if next_img.shape[2] == 4:
                next_img = cv2.cvtColor(next_img, cv2.COLOR_BGRA2BGR)
        elif display_type == "quote":
            single_image = images[(index + 1) % len(images)]
            next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
            quote, source = get_random_quote()
            next_img = add_quote_overlay(next_img, quote, source)
        else:
            single_image = images[(index + 1) % len(images)]
            next_img = create_single_image_with_background(single_image, frame_width, frame_height)
        
        # Update forecast and news periodically (e.g., every hour)
        if datetime.now().minute == 0:
            forecast = get_weather_forecast(api_key)

        frame_with_overlay = add_time_overlay(current_img, temp, weather)
        cv2.imshow('slideshow', frame_with_overlay)
        if cv2.waitKey(display_time * 1000) == ord('q'):
            cv2.destroyAllWindows()
            exit()

        transition = random.choice(transitions)
        for frame in transition(current_img, next_img, num_transition_frames):
            frame_with_overlay = add_time_overlay(frame, temp, weather)
            cv2.imshow('slideshow', frame_with_overlay)
            if cv2.waitKey(1) == ord('q'):
                cv2.destroyAllWindows()
                exit()

        frame_with_overlay = add_time_overlay(next_img, temp, weather)
        cv2.imshow('slideshow', frame_with_overlay)
        if cv2.waitKey(display_time * 1000) == ord('q'):
            cv2.destroyAllWindows()
            exit()

        current_img = next_img
        index = (index + 1) % len(images)

if __name__ == '__main__':
    main()
