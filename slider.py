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
import pillow_heif
import textwrap
import re
from collections import defaultdict

# Constants
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

def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents"
    results = service.files().list(q=query, pageSize=100, fields="nextPageToken, files(id, name, modifiedTime, size)").execute()
    return results.get('files', [])

def download_file(service, file_id, file_name):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

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

# Main Slideshow Functions
def start_slideshow(images):
    random.shuffle(images)
    cv2.namedWindow('slideshow', cv2.WINDOW_FREERATIO)
    transitions = [fade_transition, slide_transition_left, slide_transition_right, wipe_transition_top, wipe_transition_bottom]
    index = 0
    current_img = resize_and_pad(images[index], frame_width, frame_height)
    forecast = get_weather_forecast(api_key)
    news = get_ai_generated_news()

    while True:
        temp, weather = get_weather_data(api_key)
        display_type = random.choice(["single", "stitch", "quote", "forecast", "news"])
        next_img = prepare_next_frame(display_type, images, index, forecast, news)
        if datetime.now().minute == 0:
            forecast = get_weather_forecast(api_key)
            news = get_ai_generated_news()
        show_frame_with_overlay(current_img, temp, weather)
        apply_transition(current_img, next_img, temp, weather, random.choice(transitions))
        current_img = next_img
        index = (index + 1) % len(images)

def prepare_next_frame(display_type, images, index, forecast, news):
    single_image = images[(index + 1) % len(images)]
    if display_type == "forecast":
        next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
        next_img = add_forecast_overlay(next_img, forecast)
    elif display_type == "news":
        next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
        next_img = add_news_overlay(next_img, news)
    elif display_type == "stitch":
        next_img = stitch_images(random.sample(images, random.randint(2, 4)), frame_width, frame_height)
    elif display_type == "quote":
        next_img = create_zoomed_blurred_background(single_image, frame_width, frame_height)
        quote, source = get_random_quote()
        next_img = add_quote_overlay(next_img, quote, source)
    else:
        next_img = create_single_image_with_background(single_image, frame_width, frame_height)
    return next_img

def show_frame_with_overlay(frame, temp, weather):
    frame_with_overlay = add_time_overlay(frame, temp, weather)
    cv2.imshow('slideshow', frame_with_overlay)
    if cv2.waitKey(display_time * 1000) == ord('q'):
        cv2.destroyAllWindows()
        exit()

def apply_transition(current_img, next_img, temp, weather, transition):
    for frame in transition(current_img, next_img, num_transition_frames):
        frame_with_overlay = add_time_overlay(frame, temp, weather)
        cv2.imshow('slideshow', frame_with_overlay)
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            exit()

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
