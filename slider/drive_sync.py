"""Google Drive authentication, file listing, download, and media sync.

The Google client libraries are optional: without them the slideshow still
runs from the local images/ cache.
"""

import io
import os
from datetime import datetime

from slider import config
from slider.utils import read_json, write_json_atomic

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
    GOOGLE_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on the environment
    GOOGLE_AVAILABLE = False
    _IMPORT_ERROR = exc
    HttpError = Exception

MEDIA_EXTENSIONS = config.IMAGE_EXTENSIONS + config.VIDEO_EXTENSIONS
PARTIAL_SUFFIX = ".part"


def media_type_for(file_name):
    """Return "image", "video", or None for a file name."""
    ext = os.path.splitext(file_name)[1].lower()
    if ext in config.IMAGE_EXTENSIONS:
        return "image"
    if ext in config.VIDEO_EXTENSIONS:
        return "video"
    return None


def make_item(path, name, modified_time=None):
    return {
        "type": media_type_for(name),
        "path": path,
        "display_path": path,
        "name": name,
        "modifiedTime": modified_time,
    }


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate_drive():
    """Return a Drive service, or None when credentials/libraries are unavailable."""
    if not GOOGLE_AVAILABLE:
        print(f"Google Drive libraries not installed ({_IMPORT_ERROR}); using local media only.")
        return None

    creds = None
    token_path = config.TOKEN_FILE
    creds_path = config.CREDENTIALS_FILE

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
        if not creds or not creds.valid:
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
            with open(token_path, "w", encoding="utf-8") as token:
                token.write(creds.to_json())
        except OSError as exc:
            print(f"Failed to save credentials: {exc}")

    try:
        return build("drive", "v3", credentials=creds, cache_discovery=False)
    except Exception as exc:
        print(f"Failed to build Google Drive service: {exc}")
        return None


# ---------------------------------------------------------------------------
# Listing and download
# ---------------------------------------------------------------------------

def list_files_in_folder(service, folder_id):
    """List every non-trashed file in a folder (all pages). None on error."""
    if service is None:
        return None
    query = f"'{folder_id}' in parents and trashed = false"
    files = []
    page_token = None
    try:
        while True:
            request = service.files().list(
                q=query,
                pageSize=1000,
                fields="nextPageToken, files(id, name, modifiedTime, size)",
                pageToken=page_token,
            )
            results = request.execute(num_retries=2)
            files.extend(results.get("files", []))
            page_token = results.get("nextPageToken")
            if not page_token:
                break
    except HttpError as exc:
        print(f"Failed to list files: {exc}")
        return None
    except Exception as exc:
        print(f"Unexpected error listing files: {exc}")
        return None
    return files


def download_file(service, file_id, file_path):
    """Download a file to file_path atomically (via a .part temp file)."""
    if service is None:
        return False
    tmp_path = file_path + PARTIAL_SUFFIX
    try:
        request = service.files().get_media(fileId=file_id)
        with io.FileIO(tmp_path, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=2)
        os.replace(tmp_path, file_path)
        return True
    except HttpError as exc:
        print(f"Failed to download file {file_id}: {exc}")
    except OSError as exc:
        print(f"Failed to write file {file_path}: {exc}")
    except Exception as exc:
        print(f"Unexpected error downloading file {file_id}: {exc}")
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return False


def parse_modified_time(modified_time_str):
    """Parse a Drive modifiedTime string to a datetime (datetime.min on failure)."""
    if not modified_time_str:
        return datetime.min
    try:
        text = modified_time_str
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).replace(tzinfo=None)
    except ValueError:
        return datetime.min


# ---------------------------------------------------------------------------
# Local metadata and cache
# ---------------------------------------------------------------------------

def load_local_metadata(metadata_file):
    data = read_json(metadata_file)
    return data if isinstance(data, dict) else {}


def save_local_metadata(metadata_file, metadata):
    write_json_atomic(metadata_file, metadata)


def load_media_from_local_cache(temp_dir):
    """Media items already on disk, newest first (by metadata, then mtime)."""
    if not os.path.isdir(temp_dir):
        return []

    metadata = load_local_metadata(config.METADATA_FILE)
    items = []
    for entry in os.listdir(temp_dir):
        if entry.endswith(PARTIAL_SUFFIX) or media_type_for(entry) is None:
            continue
        file_path = os.path.join(temp_dir, entry)
        if not os.path.isfile(file_path):
            continue
        modified = (metadata.get(entry) or {}).get("modifiedTime")
        items.append(make_item(file_path, entry, modified))

    def sort_key(item):
        stamp = parse_modified_time(item["modifiedTime"])
        if stamp == datetime.min:
            try:
                stamp = datetime.fromtimestamp(os.path.getmtime(item["path"]))
            except OSError:
                pass
        return stamp

    items.sort(key=sort_key, reverse=True)
    return items


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def refresh_media_items(service, folder_id, temp_dir, metadata_file, local_metadata):
    """Sync the Drive folder into temp_dir.

    Returns (media_items, updated_metadata, downloaded_names), or
    (None, local_metadata, set()) when the listing failed. Local files that
    disappeared from Drive are deleted.
    """
    if service is None:
        return None, local_metadata, set()

    files = list_files_in_folder(service, folder_id)
    if files is None:
        print("Skipping media refresh due to retrieval error.")
        return None, local_metadata, set()

    media_files = [f for f in files if f.get("name") and media_type_for(f["name"])]
    if not media_files:
        return [], local_metadata, set()

    os.makedirs(temp_dir, exist_ok=True)
    media_files.sort(key=lambda f: parse_modified_time(f.get("modifiedTime")), reverse=True)

    updated_metadata = {}
    media_items = []
    downloaded = set()
    seen_names = set()

    for file in media_files:
        name = file["name"]
        if name in seen_names:  # duplicate names on Drive: keep the newest only
            continue
        seen_names.add(name)

        file_path = os.path.join(temp_dir, name)
        remote_meta = {
            "id": file.get("id"),
            "modifiedTime": file.get("modifiedTime"),
            "size": int(file.get("size", 0) or 0),
        }
        local_meta = local_metadata.get(name) or {}
        needs_download = (
            not os.path.exists(file_path)
            or local_meta.get("modifiedTime") != remote_meta["modifiedTime"]
            or int(local_meta.get("size", 0) or 0) != remote_meta["size"]
        )

        if needs_download:
            print(f"Downloading file: {name}")
            if not download_file(service, file["id"], file_path):
                print(f"Skipping file due to download error: {name}")
                if os.path.exists(file_path) and local_meta:
                    # Keep showing the previous copy until the download succeeds.
                    updated_metadata[name] = local_meta
                    media_items.append(make_item(file_path, name, local_meta.get("modifiedTime")))
                continue
            downloaded.add(name)

        media_items.append(make_item(file_path, name, remote_meta["modifiedTime"]))
        updated_metadata[name] = remote_meta

    # Remove local files that are no longer on Drive.
    for name in list(local_metadata):
        if name in updated_metadata:
            continue
        stale_path = os.path.join(temp_dir, name)
        try:
            if os.path.isfile(stale_path):
                os.remove(stale_path)
                print(f"Removed file no longer on Drive: {name}")
        except OSError as exc:
            print(f"Failed to remove {name}: {exc}")

    save_local_metadata(metadata_file, updated_metadata)
    return media_items, updated_metadata, downloaded
