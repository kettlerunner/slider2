import os
import requests
import hashlib
import subprocess

# Replace with your actual GitHub raw content URL
GITHUB_RAW_URL = "https://raw.githubusercontent.com/kettlerunner/slider2/main/slider.py"
LOCAL_SCRIPT_PATH = "slider.py"

def get_remote_script_hash(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error on bad status
    return hashlib.sha256(response.content).hexdigest()

def get_local_script_hash(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def download_script(url, file_path):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error on bad status
    with open(file_path, 'wb') as f:
        f.write(response.content)

def main():
    try:
        remote_hash = get_remote_script_hash(GITHUB_RAW_URL)
        local_hash = get_local_script_hash(LOCAL_SCRIPT_PATH)

        if local_hash != remote_hash:
            print("Updating local script...")
            download_script(GITHUB_RAW_URL, LOCAL_SCRIPT_PATH)
            print("Update completed.")
        else:
            print("Local script is up-to-date.")

        print("Running script...")
        subprocess.run(["python", LOCAL_SCRIPT_PATH])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()