import os
import time

import cv2
import numpy as np

from slider import config
from slider import image_processing as ip


def test_resize_and_pad_letterboxes_and_converts_gray():
    out = ip.resize_and_pad(np.full((100, 400), 200, np.uint8), 800, 480)
    assert out.shape == (480, 800, 3)
    assert out[0, 0].tolist() == [0, 0, 0]          # padding row is black
    assert out[240, 400].tolist() == [200, 200, 200]  # image is centered


def test_single_image_with_background_accepts_bgra(frame):
    rgba = np.zeros((300, 400, 4), np.uint8)
    rgba[..., :3] = 255
    rgba[..., 3] = 0  # fully transparent: backdrop should show through
    out = ip.create_single_image_with_background(rgba, 800, 480)
    assert out.shape == (480, 800, 3) and out.dtype == np.uint8


def test_stitch_and_blur_shapes(frame):
    images = [frame, frame[:, :400], frame[:200], np.zeros((10, 10, 3), np.uint8)]
    out = ip.stitch_images(images, 800, 480)
    assert out.shape == (480, 800, 3)
    assert ip.stitch_images([], 800, 480).shape == (480, 800, 3)
    bg = ip.create_zoomed_blurred_background(frame[:50, :50], 800, 480)
    assert bg.shape == (480, 800, 3)


def test_read_image_handles_16bit_png(tmp_path):
    path = str(tmp_path / "deep.png")
    cv2.imwrite(path, np.full((20, 30, 3), 65535, np.uint16))
    img = ip.read_image(path)
    assert img.dtype == np.uint8 and img.shape == (20, 30, 3)
    assert img.max() == 255


def test_prepare_display_copy_scales_once(tmp_path, monkeypatch):
    src = str(tmp_path / "big.jpg")
    cv2.imwrite(src, np.random.randint(0, 255, (1500, 2000, 3), np.uint8))
    dst_dir = str(tmp_path / "scaled")
    monkeypatch.setattr(config, "IMAGE_MAX_DIM", 800)

    out = ip.prepare_display_copy(src, dst_dir)
    assert out and os.path.isfile(out)
    scaled = cv2.imread(out)
    assert max(scaled.shape[:2]) == 800 and scaled.shape[1] == 800

    first_mtime = os.path.getmtime(out)
    time.sleep(0.05)
    assert ip.prepare_display_copy(src, dst_dir) == out
    assert os.path.getmtime(out) == first_mtime  # reused, not rewritten

    ip.prune_display_copies([], dst_dir)
    assert not os.path.exists(out)


def test_image_cache_is_lru(tmp_path):
    paths = []
    for i in range(3):
        p = str(tmp_path / f"{i}.png")
        cv2.imwrite(p, np.full((8, 8, 3), i, np.uint8))
        paths.append(p)
    cache = ip.ImageCache(capacity=2)
    assert cache.get(paths[0]) is not None
    assert cache.get(paths[1]) is not None
    cache.get(paths[0])            # touch 0 so 1 becomes the oldest
    cache.get(paths[2])
    assert paths[1] not in cache._items and paths[0] in cache._items
    assert cache.get(str(tmp_path / "missing.png")) is None
