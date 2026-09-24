"""Playlist: which media item is showing, which comes next, filtered by touch mode.

Items are dicts with at least "type" ("image"/"video"), "path", and "name".
Ordering is a shuffle bag: every active item plays once before any repeats.
Refreshing the item list keeps the current item and pending order intact, so a
background Drive sync never restarts the show.
"""

import random

IMAGE_ONLY_MODES = {"news", "weather", "pictures"}


class Playlist:
    def __init__(self, items=(), mode="random"):
        self.mode = mode
        self.fallback = False
        self.current = None
        self._all = []
        self._by_path = {}
        self._active = []
        self._queue = []  # paths still to play in this bag
        self.set_items(items)

    # -- filtering ---------------------------------------------------------

    def _allowed(self, item):
        if self.mode == "video":
            return item.get("type") == "video"
        if self.mode in IMAGE_ONLY_MODES:
            return item.get("type") == "image"
        return True

    def _rebuild(self, keep_current=True):
        filtered = [item for item in self._all if self._allowed(item)]
        if not filtered and self._all:
            filtered = list(self._all)
            self.fallback = True
        else:
            self.fallback = False
        self._active = filtered
        self._by_path = {item["path"]: item for item in filtered}

        current_path = self.current["path"] if (keep_current and self.current) else None
        self.current = self._by_path.get(current_path) if current_path else None

        kept = [p for p in self._queue if p in self._by_path and p != current_path]
        seen = set(kept)
        fresh = [p for p in self._by_path if p not in seen and p != current_path]
        random.shuffle(fresh)
        for path in fresh:
            # Never ahead of the already-announced next slide (index 0).
            kept.insert(random.randint(min(1, len(kept)), len(kept)), path)
        self._queue = kept

        if self.current is None and self._active:
            self.advance()

    # -- mutation ----------------------------------------------------------

    def set_items(self, items):
        """Replace the item list. Returns True when the current item changed."""
        previous = self.current["path"] if self.current else None
        self._all = [dict(item) for item in items if item.get("path") and item.get("type")]
        self._rebuild(keep_current=True)
        return (self.current["path"] if self.current else None) != previous

    def set_mode(self, mode):
        """Switch touch mode. Returns True when the mode actually changed."""
        if mode == self.mode:
            return False
        self.mode = mode
        self._rebuild(keep_current=True)
        return True

    def remove(self, item):
        """Drop an item (e.g. an unreadable file). Returns True if it was present."""
        path = item.get("path") if item else None
        if path not in {i["path"] for i in self._all}:
            return False
        self._all = [i for i in self._all if i["path"] != path]
        if self.current and self.current["path"] == path:
            self.current = None
        self._rebuild(keep_current=True)
        return True

    # -- navigation --------------------------------------------------------

    def _refill(self):
        if self._queue:
            return
        current_path = self.current["path"] if self.current else None
        paths = [p for p in self._by_path if p != current_path]
        random.shuffle(paths)
        self._queue = paths

    def peek_next(self):
        """The item that will play after the current one, or None if there is none."""
        if len(self._active) <= 1:
            return None
        self._refill()
        return self._by_path.get(self._queue[0]) if self._queue else None

    def advance(self):
        """Move to the next item and return it (None when the playlist is empty)."""
        if not self._active:
            self.current = None
            return None
        self._refill()
        if not self._queue:  # only one item
            self.current = self._active[0]
            return self.current
        self.current = self._by_path[self._queue.pop(0)]
        return self.current

    # -- views -------------------------------------------------------------

    @property
    def active_items(self):
        return list(self._active)

    @property
    def all_items(self):
        return list(self._all)

    @property
    def image_paths(self):
        return [item.get("display_path") or item["path"] for item in self._active if item.get("type") == "image"]

    def __len__(self):
        return len(self._active)
