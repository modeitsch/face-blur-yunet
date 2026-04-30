import numpy as np

from face_blur_yunet.face_blur import BlurOptions, clamp_box, blur_face_only


def test_clamp_box_keeps_box_inside_frame():
    assert clamp_box((-10.2, 5.4, 50.1, 20.2), 100, 80) == (0, 5, 40, 20)
    assert clamp_box((90, 70, 30, 30), 100, 80) == (90, 70, 10, 10)


def test_clamp_box_returns_empty_box_when_fully_left_of_frame():
    assert clamp_box((-50, 10, 20, 20), 100, 80) == (0, 10, 0, 20)


def test_clamp_box_applies_padding_inside_frame():
    options = BlurOptions(face_padding=0.25)
    assert clamp_box((20, 20, 40, 20), 100, 80, options) == (10, 15, 60, 30)


def test_clamp_box_returns_empty_box_when_fully_outside_frame():
    assert clamp_box((120, 90, 20, 20), 100, 80) == (100, 80, 0, 0)


def test_blur_face_only_changes_pixels_inside_roi():
    frame = np.zeros((80, 100, 3), dtype=np.uint8)
    frame[20:60, 30:70] = np.indices((40, 40)).sum(axis=0)[:, :, None] % 2 * 255
    original = frame.copy()
    blur_face_only(frame, (30, 20, 40, 40), BlurOptions(blur_strength=31))
    assert np.any(frame != original)
