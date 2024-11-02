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
from PIL import Image
import textwrap
import re
from collections import defaultdict
import random

# Constants ... updated  
frame_width = 800
frame_height = 480
transition_time = 2
display_time = 10
num_transition_frames = int(transition_time * 30)
api_key = os.getenv('WEATHERMAP_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_key)
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Utility Functions
def sanitize_text(text):
    replacements = {'"': '"', '–': '-', '—': '-', '…': '...'}
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def load_local_metadata(metadata_file):
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            return json.load(f)
    return {}

def save_local_metadata(metadata_file, metadata):
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

def load_image(file_path):
    """Helper to load and convert images to BGRA format. Converts HEIC to JPEG if necessary."""
    if file_path.lower().endswith(".heic"):
        try:
            heif_file = pillow_heif.read_heif(file_path)
            image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
            jpeg_path = file_path.rsplit(".", 1)[0] + ".jpeg"
            image = image.convert("RGB")
            image.save(jpeg_path, "JPEG")
            file_path = jpeg_path
            print(f"Converted HEIC to JPEG: {file_path}")
        except Exception as e:
            print(f"Failed to convert HEIC image: {file_path}. Error: {e}")
            return None
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img

# Google Drive Functions
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

def get_random_quote():
    """Fetch a random inspirational quote using OpenAI."""
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

    try:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": chosen_prompt}]
        )

        # Extract the response content and clean it up
        quote_data = completion.choices[0].message.content.strip()

        # Ensure the response is a valid JSON string
        if quote_data.startswith('```json'):
            quote_data = quote_data[len('```json'):].strip()
        if quote_data.endswith('```'):
            quote_data = quote_data[:-len('```')].strip()

        # Parse the JSON response
        quote_json = json.loads(quote_data)
        return quote_json.get("quote", ""), quote_json.get("source", "")
    except Exception as e:
        print(f"Error generating quote: {e}")
        return "An error occurred while generating a quote.", "Error"

def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name, modifiedTime, size)").execute()
    return results.get('files', [])

def show_frame_with_overlay(frame, temp, weather):
    """Display the frame with time and weather overlay."""
    # Get the current time
    time_text = datetime.now().strftime("%B %d %Y, %I:%M %p")
    
    # Prepare weather information
    weather_text = f"Temp: {temp:.1f} F" if temp is not None else "Weather data unavailable"
    if weather:
        if "rain" in weather.lower():
            weather_text += ", Rain Predicted"
        elif "snow" in weather.lower():
            weather_text += ", Snow Predicted"
        elif "cloud" in weather.lower():
            weather_text += ", Cloudy"
        else:
            weather_text += ", Clear"

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_color = (255, 255, 255)
    thickness = 1

    # Calculate positions for time and weather text
    text_size_time, _ = cv2.getTextSize(time_text, font, font_scale, thickness)
    text_size_weather, _ = cv2.getTextSize(weather_text, font, font_scale, thickness)
    
    text_x_weather = 10
    text_y = frame.shape[0] - 20
    text_x_time = frame.shape[1] - text_size_time[0] - 10

    # Create a semi-transparent overlay
    overlay_frame = frame.copy()
    bar_height = max(text_size_time[1], text_size_weather[1]) + 20
    overlay = overlay_frame.copy()
    cv2.rectangle(overlay, (0, frame.shape[0] - bar_height), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
    alpha = 0.8  # Transparency factor
    cv2.addWeighted(overlay, alpha, overlay_frame, 1 - alpha, 0, overlay_frame)

    # Add text to the overlay frame
    cv2.putText(overlay_frame, weather_text, (text_x_weather, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
    cv2.putText(overlay_frame, time_text, (text_x_time, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('slideshow', overlay_frame)
    if cv2.waitKey(display_time * 1000) == ord('q'):
        cv2.destroyAllWindows()
        exit()

def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

def create_single_image_with_background(image, width, height):
    """Create a displayable image by adding a blurred background to a single image."""
    # Check if the image has an alpha channel
    if image.shape[2] == 4:
        bg_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bg_image = image.copy()

    # Create a blurred background
    blurred_background = create_zoomed_blurred_background(bg_image, width, height)
    resized_image = resize_and_pad(image, width, height)

    # Calculate padding to center the resized image
    top_pad = (height - resized_image.shape[0]) // 2
    left_pad = (width - resized_image.shape[1]) // 2

    # Overlay the resized image onto the blurred background
    if resized_image.shape[2] == 4:
        rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGRA2BGR)
        alpha = resized_image[:, :, 3] / 255.0
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
        background = blurred_background[top_pad:top_pad+resized_image.shape[0], left_pad:left_pad+resized_image.shape[1]]
        combined = (1 - alpha) * background + alpha * rgb_image
        blurred_background[top_pad:top_pad+resized_image.shape[0], left_pad:left_pad+resized_image.shape[1]] = combined
    else:
        blurred_background[top_pad:top_pad+resized_image.shape[0], left_pad:left_pad+resized_image.shape[1]] = resized_image

    return blurred_background

# Data Functions
def get_weather_forecast(api_key, city="Waupun", country_code="US"):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},{country_code}&units=imperial&appid={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        daily_forecast = defaultdict(lambda: {'temp_min': float('inf'), 'temp_max': float('-inf'), 'descriptions': []})
        for item in data['list']:
            dt = datetime.fromtimestamp(item['dt'])
            date_key = dt.strftime("%Y-%m-%d")
            daily_forecast[date_key]['temp_min'] = min(daily_forecast[date_key]['temp_min'], item['main']['temp_min'])
            daily_forecast[date_key]['temp_max'] = max(daily_forecast[date_key]['temp_max'], item['main']['temp_max'])
            daily_forecast[date_key]['descriptions'].append(item['weather'][0]['description'])
            if dt.hour == 12:
                daily_forecast[date_key]['main_description'] = item['weather'][0]['description']
        formatted_forecast = []
        for date, forecast in daily_forecast.items():
            dt = datetime.strptime(date, "%Y-%m-%d")
            formatted_forecast.append({
                'date': dt.strftime("%A, %b %d"),
                'temp_min': forecast['temp_min'],
                'temp_max': forecast['temp_max'],
                'description': forecast.get('main_description', max(set(forecast['descriptions']), key=forecast['descriptions'].count))
            })
        formatted_forecast.sort(key=lambda x: datetime.strptime(x['date'], "%A, %b %d"))
        return formatted_forecast[:5]
    print("Error fetching weather forecast data")
    return None

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
    print("Error fetching weather data")
    return None, None

# Transition Functions
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
        dx = int(current_img.shape[1] * i / num_frames)
        frame = np.zeros_like(current_img)
        frame[:, :current_img.shape[1] - dx] = current_img[:, dx:]
        frame[:, current_img.shape[1] - dx:] = next_img[:, :dx]
        yield frame

def slide_transition_right(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for i in range(num_frames):
        dx = int(current_img.shape[1] * i / num_frames)
        frame = np.zeros_like(current_img)
        frame[:, dx:] = current_img[:, :current_img.shape[1] - dx]
        frame[:, :dx] = next_img[:, current_img.shape[1] - dx:]
        yield frame

def wipe_transition_top(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for i in range(num_frames):
        dy = int(current_img.shape[0] * i / num_frames)
        frame = np.zeros_like(current_img)
        frame[:current_img.shape[0] - dy, :] = current_img[dy:, :]
        frame[current_img.shape[0] - dy:, :] = next_img[:dy, :]
        yield frame

def wipe_transition_bottom(current_img, next_img, num_frames):
    current_img, next_img = ensure_same_channels(current_img, next_img)
    for i in range(num_frames):
        dy = int(current_img.shape[0] * i / num_frames)
        frame = np.zeros_like(current_img)
        frame[dy:, :] = current_img[:current_img.shape[0] - dy, :]
        frame[:dy, :] = next_img[current_img.shape[0] - dy:, :]
        yield frame

# Image Processing Functions
def resize_and_pad(image, width, height):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    has_alpha = image.shape[2] == 4
    padded_image = np.zeros((height, width, 4 if has_alpha else 3), dtype=np.uint8)
    top_pad = (height - resized_image.shape[0]) // 2
    left_pad = (width - resized_image.shape[1]) // 2
    padded_image[top_pad:top_pad+resized_image.shape[0], left_pad:left_pad+resized_image.shape[1]] = resized_image
    return padded_image

def create_zoomed_blurred_background(image, width, height):
    image_copy = image.copy()
    h, w = image_copy.shape[:2]
    scale = max(width / w, height / h) * 1.1
    zoomed_image = cv2.resize(image_copy, (int(w * scale), int(h * scale)))
    start_x = (zoomed_image.shape[1] - width) // 2
    start_y = (zoomed_image.shape[0] - height) // 2
    cropped_image = zoomed_image[start_y:start_y + height, start_x:start_x + width]
    blurred_image = cv2.GaussianBlur(cropped_image, (51, 51), 0)
    return blurred_image

def stitch_images(images, width, height):
    num_images = len(images)
    stitched_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    background_image = random.choice(images)
    if background_image.shape[2] == 4:
        background_image = cv2.cvtColor(background_image, cv2.COLOR_BGRA2BGR)
    blurred_background = create_zoomed_blurred_background(background_image, width, height)
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
            alpha = resized_image[:, :, 3] / 255.0
            alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)
            stitched_image[start_y:start_y+grid_height, start_x:start_x+grid_width] = (
                (1 - alpha) * stitched_image[start_y:start_y+grid_height, start_x:start_x+grid_width] +
                alpha * rgb_image
            )
        else:
            stitched_image[start_y:start_y+grid_height, start_x:start_x+grid_width] = resized_image
    return stitched_image

# Main Slideshow Functions
def start_slideshow(images):
    random.shuffle(images)
    cv2.namedWindow('slideshow', cv2.WINDOW_FREERATIO)
    transitions = [fade_transition, slide_transition_left, slide_transition_right, wipe_transition_top, wipe_transition_bottom]
    index = 0
    current_img = resize_and_pad(images[index], frame_width, frame_height)
    forecast = get_weather_forecast(api_key)

    while True:
        temp, weather = get_weather_data(api_key)
        display_type = random.choice(["single", "stitch", "quote", "forecast"])
        next_img = prepare_next_frame(display_type, images, index, forecast)
        if datetime.now().minute == 0:
            forecast = get_weather_forecast(api_key)
            
        show_frame_with_overlay(current_img, temp, weather)
        apply_transition(current_img, next_img, temp, weather, random.choice(transitions))
        current_img = next_img
        index = (index + 1) % len(images)

def prepare_next_frame(display_type, images, index, forecast):
    single_image = images[(index + 1) % len(images)]
    if display_type == "forecast":
        next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
        next_img = add_forecast_overlay(next_img, forecast)
    elif display_type == "stitch":
        next_img = stitch_images(random.sample(images, min(4, len(images))), frame_width, frame_height)
    elif display_type == "quote":
        next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
        quote, source = get_random_quote()
        next_img = add_quote_overlay(next_img, quote, source)
    else:
        next_img = create_single_image_with_background(single_image, frame_width, frame_height)
    return next_img

# Main Program
def main():
    service = authenticate_drive()
    folder_id = '1hpBzZ_kiXpIBtRv1FN3da8zOhT5J0Ggi'
    files = list_files_in_folder(service, folder_id)
    temp_dir = 'images'
    os.makedirs(temp_dir, exist_ok=True)
    metadata_file = 'metadata.json'
    local_metadata = load_local_metadata(metadata_file)
    images = []

    for file in files:
        file_name = file['name']
        file_path = os.path.join(temp_dir, file_name)
        file_metadata = {'name': file_name, 'modifiedTime': file.get('modifiedTime'), 'size': file.get('size', 0)}
        if file_name in local_metadata:
            local_file_metadata = local_metadata[file_name]
            if (local_file_metadata['modifiedTime'] == file_metadata['modifiedTime'] and 
                local_file_metadata['size'] == file_metadata['size'] and 
                os.path.exists(file_path)):
                print(f"Skipping download of unchanged file: {file_name}")
                img = load_image(file_path)
                if img is not None:
                    images.append(img)
                continue
        print(f"Downloading file: {file_name}")
        download_file(service, file['id'], file_path)
        img = load_image(file_path)
        if img is not None:
            images.append(img)
            local_metadata[file_name] = file_metadata
            save_local_metadata(metadata_file, local_metadata)

    if not images:
        print("No images found in the folder.")
        return

    start_slideshow(images)

if __name__ == '__main__':
    main()
