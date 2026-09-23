from slider.playlist import Playlist


def item(name, kind="image"):
    return {"type": kind, "path": f"/media/{name}", "display_path": f"/media/{name}", "name": name}


def test_bag_plays_everything_before_repeating():
    items = [item(f"{i}.jpg") for i in range(6)]
    playlist = Playlist(items)
    seen = {playlist.current["path"]}
    for _ in range(5):
        nxt = playlist.peek_next()
        assert playlist.advance() is nxt
        seen.add(nxt["path"])
    assert len(seen) == 6


def test_mode_filters_and_falls_back():
    items = [item("a.jpg"), item("b.mp4", "video"), item("c.jpg")]
    playlist = Playlist(items)
    assert playlist.set_mode("video") and len(playlist) == 1 and not playlist.fallback
    assert playlist.current["type"] == "video"
    assert playlist.set_mode("news") and len(playlist) == 2
    assert all(i["type"] == "image" for i in playlist.active_items)
    assert not playlist.set_mode("news")  # unchanged

    only_images = Playlist([item("a.jpg")], mode="video")
    assert only_images.fallback and len(only_images) == 1


def test_refresh_keeps_current_and_pending_order():
    items = [item(f"{i}.jpg") for i in range(4)]
    playlist = Playlist(items)
    current = playlist.current["path"]
    upcoming = playlist.peek_next()["path"]
    changed = playlist.set_items(items + [item("new.jpg")])
    assert not changed
    assert playlist.current["path"] == current
    assert playlist.peek_next()["path"] == upcoming
    paths = {i["path"] for i in playlist.active_items}
    assert "/media/new.jpg" in paths and len(paths) == 5


def test_refresh_replaces_item_dicts_for_updated_display_path():
    playlist = Playlist([item("a.jpg"), item("b.jpg")])
    updated = [dict(i, display_path=i["path"] + ".scaled") for i in playlist.all_items]
    playlist.set_items(updated)
    assert playlist.current["display_path"].endswith(".scaled")


def test_remove_current_moves_on_and_empty_playlist_is_safe():
    playlist = Playlist([item("a.jpg"), item("b.jpg")])
    current = playlist.current
    assert playlist.remove(current)
    assert playlist.current is not None and playlist.current["path"] != current["path"]
    assert playlist.peek_next() is None  # only one left
    assert playlist.remove(playlist.current)
    assert playlist.current is None and playlist.advance() is None and len(playlist) == 0
    assert not playlist.remove({"path": "/nope"})
