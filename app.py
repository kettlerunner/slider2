import os
import requests
import hashlib
import subprocess

# Base URL for your GitHub raw content
BASE_GITHUB_RAW_URL = "https://raw.githubusercontent.com/kettlerunner/slider2/main/"

# List of files to update: (remote_file_name, local_file_path)
FILES_TO_UPDATE = [
    ("slider.py", "slider.py"),
    ("quotes.json", "quotes.json")
]

def get_remote_file_hash_and_content(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an error on bad status
    content = response.content
    file_hash = hashlib.sha256(content).hexdigest()
    return file_hash, content

def get_local_file_hash(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def download_file(url, file_path, content=None):
    if content is None:
        response = requests.get(url)
        response.raise_for_status()
        content = response.content
    with open(file_path, 'wb') as f:
        f.write(content)

def update_file(remote_file_name, local_file_path):
    url = BASE_GITHUB_RAW_URL + remote_file_name
    remote_hash, remote_content = get_remote_file_hash_and_content(url)
    local_hash = get_local_file_hash(local_file_path)

    if local_hash != remote_hash:
        print(f"Updating {local_file_path}...")
        download_file(url, local_file_path, content=remote_content)
        print(f"{local_file_path} has been updated.")
    else:
        print(f"{local_file_path} is up-to-date.")

def main():
    try:
        for remote_file_name, local_file_path in FILES_TO_UPDATE:
            update_file(remote_file_name, local_file_path)
        
        print("Running slider.py...")
        subprocess.run(["python", "slider.py"])
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
