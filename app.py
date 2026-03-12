"""Slider2 launcher — pulls updates from git, then runs the slideshow package."""

import os
import subprocess
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GIT_TIMEOUT = 15  # seconds
RETRY_DELAY = 10  # seconds


def git_pull():
    """Attempt a fast-forward git pull. All failures are non-fatal."""
    try:
        result = subprocess.run(
            ["git", "pull", "--ff-only"],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT,
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            if output:
                print(f"Update: {output}")
        else:
            print(f"git pull failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("git not found. Skipping update check.")
    except subprocess.TimeoutExpired:
        print("git pull timed out. Running with current code.")
    except Exception as exc:
        print(f"Update check failed: {exc}")


def main():
    git_pull()
    print("Starting slideshow...")

    while True:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "slider"],
                cwd=SCRIPT_DIR,
            )
            if result.returncode == 0:
                print("Slideshow exited cleanly.")
                break
        except KeyboardInterrupt:
            print("Received interrupt. Shutting down.")
            break
        except Exception as exc:
            print(f"Slideshow crashed: {exc}")

        print(f"Restarting in {RETRY_DELAY}s...")
        time.sleep(RETRY_DELAY)
        git_pull()  # Check for updates before restart


if __name__ == "__main__":
    main()
