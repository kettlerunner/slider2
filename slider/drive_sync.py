"""Google Drive authentication, file listing, download, and media sync."""

import io
import json
import os
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

from slider import config


def authenticate_drive():
    """Authenticate to Google Drive using credentials/token stored next to script."""
    creds = None
    token_path = config.resource_path("token.json")
    creds_path = config.resource_path("credentials.json")

    if os.path.exists(token_path):
        try:
            creds = Credentials.from_authorized_user_file(token_path, config.DRIVE_SCOPES)
        except Exception as exc:
            print(f"Failed to load existing credentials: {exc}")
            creds = None

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                print(f"Failed to refresh credentials: {exc}")
                creds = None
        else:
            if not os.path.exists(creds_path):
                print(f"Missing credentials.json at {creds_path}.")
                return None
            try:
                flow = InstalledAppFlow.from_client_secrets_file(creds_path, config.DRIVE_SCOPES)
                creds = flow.run_local_server(port=0)
            except Exception as exc:
                print(f"Failed to authenticate with Google Drive: {exc}")
                return None
        try:
            with open(token_path, "w") as token:
                token.write(creds.to_json())
        except OSError as exc:
            print(f"Failed to save credentials: {exc}")

    try:
        return build("drive", "v3", credentials=creds)
    except Exception as exc:
        print(f"Failed to build Google Drive service: {exc}")
        return None


def list_files_in_folder(service, folder_id):
    """List files in a Google Drive folder."""
    if service is None:
        return None
    query = f"'{folder_id}' in parents"
    try:
        results = (
            service.files()
            .list(q=query, pageSize=200, fields="nextPageToken, files(id, name, modifiedTime, size)")
            .execute()
        )
    except HttpError as exc:
        print(f"Failed to list files: {exc}")
        return None
    except Exception as exc:
        print(f"Unexpected error listing files: {exc}")
        return None
    return results.get("files", [])


def download_file(service, file_id, file_path):
    """Download a single file from Google Drive."""
    if service is None:
        return False
    try:
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(file_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
    except HttpError as exc:
        print(f"Failed to download file {file_id}: {exc}")
        return False
    except OSError as exc:
        print(f"Failed to write file {file_path}: {exc}")
        return False
    except Exception as exc:
        print(f"Unexpected error downloading file {file_id}: {exc}")
        return False
    return True


def parse_modified_time(modified_time_str):
    """Parse a Google Drive modifiedTime string to datetime."""
    if not modified_time_str:
        return datetime.min
    try:
        if modified_time_str.endswith("Z"):
            modified_time_str = modified_time_str[:-1] + "+00:00"
        return datetime.fromisoformat(modified_time_str)
    except ValueError:
        return datetime.min


def load_local_metadata(metadata_file):
    """Load metadata dict from a JSON file."""
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Failed to load metadata: {exc}")
    return {}


def save_local_metadata(metadata_file, metadata):
    """Save metadata dict to a JSON file."""
    try:
        with open(metadata_file, "w") as f:
            json.dump(metadata, f)
    except OSError as exc:
        print(f"Failed to save metadata: {exc}")


def load_media_from_local_cache(temp_dir):
    """Load media items that were previously downloaded to disk (paths only)."""
    if not os.path.isdir(temp_dir):
        return []

    media_items = []
    for entry in sorted(os.listdir(temp_dir)):
        file_path = os.path.join(temp_dir, entry)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(entry)[1].lower()
        if ext in config.IMAGE_EXTENSIONS:
            media_type = "image"
        elif ext in config.VIDEO_EXTENSIONS:
            media_type = "video"
        else:
            continue

        media_items.append({
            "type": media_type,
            "path": file_path,
            "name": entry,
            "modifiedTime": None,
        })

    return media_items


def refresh_media_items(service, folder_id, temp_dir, metadata_file, local_metadata):
    """Sync media from Google Drive.

    Returns (media_items, updated_metadata, downloaded_files) or
    (None, local_metadata, set()) on error.
    """
    if service is None:
        return None, local_metadata, set()

    files = list_files_in_folder(service, folder_id)
    if files is None:
        print("Skipping media refresh due to retrieval error.")
        return None, local_metadata, set()

    if not files:
        save_local_metadata(metadata_file, {})
        return [], {}, set()

    os.makedirs(temp_dir, exist_ok=True)

    sorted_files = sorted(
        files,
        key=lambda item: parse_modified_time(item.get("modifiedTime")),
        reverse=True,
    )

    updated_metadata = {}
    media_items = []
    downloaded_files = set()

    for file in sorted_files:
        file_name = file["name"]
        file_path = os.path.join(temp_dir, file_name)
        remote_size = int(file.get("size", 0) or 0)
        file_metadata = {
            "modifiedTime": file.get("modifiedTime"),
            "size": remote_size,
        }

        ext = os.path.splitext(file_name)[1].lower()
        if ext not in config.IMAGE_EXTENSIONS + config.VIDEO_EXTENSIONS:
            continue

        needs_download = False
        local_file_metadata = local_metadata.get(file_name)
        if local_file_metadata is None:
            needs_download = True
        else:
            local_size = int(local_file_metadata.get("size", 0) or 0)
            if (
                local_file_metadata.get("modifiedTime") != file_metadata["modifiedTime"]
                or local_size != file_metadata["size"]
            ):
                needs_download = True
            elif not os.path.exists(file_path):
                needs_download = True

        if needs_download:
            print(f"Downloading file: {file_name}")
            if not download_file(service, file["id"], file_path):
                print(f"Skipping file due to download error: {file_name}")
                continue
            downloaded_files.add(file_name)

        if not os.path.exists(file_path):
            print(f"File missing after download: {file_name}")
            continue

        media_type = "video" if ext in config.VIDEO_EXTENSIONS else "image"
        media_items.append({
            "type": media_type,
            "path": file_path,
            "name": file_name,
            "modifiedTime": file_metadata["modifiedTime"],
        })

        updated_metadata[file_name] = file_metadata

    save_local_metadata(metadata_file, updated_metadata)
    return media_items, updated_metadata, downloaded_files
