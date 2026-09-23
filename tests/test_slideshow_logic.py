from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np

from slider import config
from slider.demo import demo_five_day, demo_news, demo_summary, demo_weather
from slider.playlist import Playlist
from slider.services import SharedState
from slider.slideshow import SlideBuilder, choose_display_type, message_frame


def _snapshot(**overrides):
    state = SharedState()
    state.update(weather=demo_weather(), forecast_5day=demo_five_day(),
                 forecast_summary=demo_summary(), news_pool=demo_news(), news_fetched_at=datetime.now())
    state.update(**overrides)
    return state.snapshot()


def test_choose_display_type_skips_missing_data():
    weekday_noon = datetime(2026, 9, 23, 12, 0)
    snap = _snapshot(forecast_5day=None, forecast_summary=None, news_pool=[])
    for _ in range(50):
        assert choose_display_type(snap, weekday_noon) in {"single", "stitch", "quote"}
    snap = _snapshot()
    evening = datetime(2026, 9, 23, 22, 0)
    for _ in range(50):
        assert choose_display_type(snap, evening) in set(config.DISPLAY_MIX["evening"])
    assert choose_display_type(snap, weekday_noon, weights={"nothing": 0}) == "single"


def test_slide_builder_covers_every_layout(tmp_path):
    paths = []
    for i in range(3):
        p = str(tmp_path / f"{i}.jpg")
        cv2.imwrite(p, np.random.randint(0, 255, (600, 900, 3), np.uint8))
        paths.append(p)
    items = [{"type": "image", "path": p, "display_path": p, "name": f"{i}.jpg"} for i, p in enumerate(paths)]
    playlist = Playlist(items)
    builder = SlideBuilder()
    snap = _snapshot()
    item = playlist.current

    for mode in ("news", "weather", "pictures", "random"):
        for _ in range(4):
            frame = builder.build(item, playlist.image_paths, snap, mode)
            assert frame.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3) and frame.dtype == np.uint8
    for layout in ("forecast", "stitch", "quote", "today", "news"):
        frame = getattr(builder, f"{layout}_frame")(builder._load(item), *(
            [item, playlist.image_paths] if layout == "stitch" else [snap] if layout != "quote" else []))
        assert frame.shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)

    missing = {"type": "image", "path": str(tmp_path / "nope.jpg"), "display_path": str(tmp_path / "nope.jpg"), "name": "nope"}
    assert builder.build(missing, [], snap, "random") is None
    assert builder.build({"type": "video", "path": paths[0], "name": "v"}, [], snap, "random") is None
    assert message_frame("Waiting...").shape == (config.FRAME_HEIGHT, config.FRAME_WIDTH, 3)


def test_shared_state_snapshot_is_isolated():
    state = SharedState()
    state.publish_media([{"path": "a"}])
    snap = state.snapshot()
    assert snap.media_version == 1 and snap.media_items == [{"path": "a"}]
    state.publish_media([])
    assert snap.media_items == [{"path": "a"}]  # snapshot unaffected by later writes
    assert isinstance(snap, SimpleNamespace)
