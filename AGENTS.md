# Agent Instructions

This project runs on a Raspberry Pi 4 using the current Raspberry Pi OS and the official 7-inch Raspberry Pi touchscreen (800x480). Prioritize solutions that remain lightweight and robust on this hardware. Ensure the slideshow keeps running continuously even when network connections are unreliable.

## Architecture

The application is a Python package (`slider/`) launched via `app.py` or `python -m slider`. Key modules:

- `config.py` — All settings loaded from `config.json` + env vars. No hardcoded values elsewhere.
- `ai_client.py` — OpenAI wrapper (Responses API primary, Chat Completions secondary).
- `slideshow.py` — Main loop orchestrator. Imports all other modules.
- `transitions.py` — Melt and wave transitions are vectorized with NumPy/cv2.remap for Pi performance.

## Rules

- Keep all configuration in `config.json` / `config.py`. Never hardcode cities, API keys, folder IDs, or display dimensions in other modules.
- OpenAI calls go through `ai_client.py` — never call the SDK directly from other modules.
- All network operations must fail gracefully (return None / fallback values). The slideshow must never crash due to network issues.
- Test with `python -m pytest tests/` from the project root.
