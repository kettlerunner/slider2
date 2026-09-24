import numpy as np
import pytest

from slider import config


@pytest.fixture
def frame():
    """A frame-sized BGR gradient image."""
    h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
    ys, xs = np.indices((h, w), dtype=np.float32)
    img = np.stack([xs / w * 255, ys / h * 255, np.full((h, w), 128, np.float32)], axis=-1)
    return img.astype(np.uint8)


@pytest.fixture
def frame2():
    h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
    return np.random.RandomState(1).randint(0, 255, (h, w, 3)).astype(np.uint8)


@pytest.fixture
def isolated_caches(tmp_path, monkeypatch):
    """Point every on-disk cache at a temp directory."""
    for name in ("WEATHER_CACHE_FILE", "FORECAST_CACHE_FILE", "NEWS_CACHE_FILE", "METADATA_FILE"):
        monkeypatch.setattr(config, name, str(tmp_path / name.lower()))
    monkeypatch.setattr(config, "IMAGES_DIR", str(tmp_path / "images"))
    monkeypatch.setattr(config, "SCALED_DIR", str(tmp_path / "images" / ".scaled"))
    return tmp_path
