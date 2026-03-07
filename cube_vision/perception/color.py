"""Color-based object detection using HSV thresholding."""

from dataclasses import dataclass
from pathlib import Path
import json

import cv2
import numpy as np

from cube_vision.config.paths import REALSENSE_OUTPUT_DIR


@dataclass
class Detection:
    centroid_px: tuple[int, int]
    area: float
    bbox: tuple[int, int, int, int]
    contour: np.ndarray


COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red": [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "green": [((35, 80, 50), (85, 255, 255))],
    "blue": [((100, 120, 50), (130, 255, 255))],
}


def detect_color(
    bgr: np.ndarray,
    color: str,
    min_area: int = 100,
    blur_ksize: int = 5,
    exclude_bottom_fraction: float = 0.0,
) -> list[Detection]:
    key = color.lower()
    if key not in COLOR_RANGES:
        raise ValueError(f"Unknown color {color!r}. Choose from: {', '.join(COLOR_RANGES)}")

    ranges = COLOR_RANGES[key]
    blurred = cv2.GaussianBlur(bgr, (blur_ksize, blur_ksize), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, np.array(ranges[0][0]), np.array(ranges[0][1]))
    for low, high in ranges[1:]:
        mask |= cv2.inRange(hsv, np.array(low), np.array(high))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if exclude_bottom_fraction > 0.0:
        h = mask.shape[0]
        cutoff = int(h * (1.0 - exclude_bottom_fraction))
        mask[max(0, min(h, cutoff)) :, :] = 0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[Detection] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        moments = cv2.moments(cnt)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        detections.append(Detection((cx, cy), area, (x, y, w, h), cnt))

    detections.sort(key=lambda det: det.area, reverse=True)
    return detections


def detection_to_xyz(
    det: Detection,
    depth_m: np.ndarray,
    intrinsics: dict,
    patch: int = 5,
) -> np.ndarray:
    cx, cy = det.centroid_px
    h, w = depth_m.shape
    half = patch // 2
    y0 = max(0, cy - half)
    y1 = min(h, cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(w, cx + half + 1)
    region = depth_m[y0:y1, x0:x1]
    valid = region[(region > 0.05) & (region < 1.5)]
    if valid.size == 0:
        raise RuntimeError(f"No valid depth at pixel ({cx}, {cy})")

    z = float(np.median(valid))
    x_m = (cx - intrinsics["ppx"]) * z / intrinsics["fx"]
    y_m = (cy - intrinsics["ppy"]) * z / intrinsics["fy"]
    return np.array([x_m, y_m, z])


def detect_object(
    color: str = "red",
    captures_dir: str | Path | None = None,
    exclude_bottom_fraction: float = 0.0,
) -> np.ndarray:
    captures_dir = REALSENSE_OUTPUT_DIR if captures_dir is None else Path(captures_dir)
    bgr = cv2.imread(str(captures_dir / "color.png"))
    if bgr is None:
        raise FileNotFoundError(f"color.png not found in {captures_dir}")
    depth_m = np.load(captures_dir / "depth_meters.npy")
    with open(captures_dir / "intrinsic_data.json") as f:
        intrinsics = json.load(f)

    detections = detect_color(bgr, color, exclude_bottom_fraction=exclude_bottom_fraction)
    if not detections:
        raise RuntimeError(f"No {color} objects detected in image")

    best = detections[0]
    print(
        f"Color detection: found {len(detections)} {color} blob(s). "
        f"Largest at pixel {best.centroid_px}, area={best.area:.0f}"
    )
    centroid_3d = detection_to_xyz(best, depth_m, intrinsics)
    print(f"3-D centroid (camera optical frame): {centroid_3d}")
    return centroid_3d
