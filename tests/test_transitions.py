import numpy as np
import pytest

from slider import transitions


@pytest.mark.parametrize("transition", transitions.ALL_TRANSITIONS)
def test_transition_endpoints_and_shape(transition, frame, frame2):
    render = transition(frame, frame2)
    start = render(0.0)
    end = render(1.0)
    mid = render(0.5)
    assert start.shape == frame.shape and start.dtype == np.uint8
    assert np.array_equal(start, frame), transition.__name__
    assert np.array_equal(end, frame2), transition.__name__
    assert mid.shape == frame.shape
    # The midpoint must actually be a mix, not one of the endpoints.
    assert not np.array_equal(mid, frame) and not np.array_equal(mid, frame2)


@pytest.mark.parametrize("transition", transitions.ALL_TRANSITIONS)
def test_alpha_is_clamped(transition, frame, frame2):
    render = transition(frame, frame2)
    assert np.array_equal(render(-0.5), frame)
    assert np.array_equal(render(1.5), frame2)


def test_iter_frames_count(frame, frame2):
    frames = list(transitions.iter_frames(transitions.fade, frame, frame2, 7))
    assert len(frames) == 7
    assert np.array_equal(frames[0], frame) and np.array_equal(frames[-1], frame2)


def test_generator_aliases_exist(frame, frame2):
    frames = list(transitions.wave_transition(frame, frame2, 3))
    assert len(frames) == 3


def test_prepare_handles_mismatched_inputs(frame):
    gray = np.zeros((100, 200), dtype=np.uint8)
    bgra = np.zeros((50, 60, 4), dtype=np.uint8)
    cur, nxt = transitions.prepare(gray, bgra)
    assert cur.shape == (100, 200, 3) and nxt.shape == (100, 200, 3)
    render = transitions.slide_left(cur, nxt)
    assert render(0.3).shape == (100, 200, 3)
