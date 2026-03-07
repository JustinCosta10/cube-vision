import numpy as np

from cube_vision.perception.color import detect_color


def test_detect_color_finds_largest_blob():
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[10:30, 10:30] = (0, 0, 255)
    img[40:75, 40:75] = (0, 0, 255)

    detections = detect_color(img, "red")

    assert len(detections) == 2
    assert detections[0].centroid_px[0] > detections[1].centroid_px[0]


def test_detect_color_can_exclude_bottom_region():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[85:98, 20:40] = (0, 255, 0)

    detections = detect_color(img, "green", exclude_bottom_fraction=0.20)

    assert detections == []
