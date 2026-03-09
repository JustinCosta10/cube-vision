"""Color-based object detection and point cloud processing."""

import json
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from cube_vision import REALSENSE_OUTPUT_DIR

# ---------------------------------------------------------------------------
# Color detection
# ---------------------------------------------------------------------------


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
        mask[max(0, min(h, cutoff)):, :] = 0

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


# ---------------------------------------------------------------------------
# Point cloud
# ---------------------------------------------------------------------------


class PointCloud:
    def __init__(self, captures_dir=None):
        self.captures_dir = REALSENSE_OUTPUT_DIR if captures_dir is None else Path(captures_dir)
        self.ply_path = self.captures_dir / "vision.ply"
        self.pcd = None
        self.inlier_cloud = None
        self.outlier_cloud = None
        self.plane_model = None

    def load_from_ply(self, ply_path):
        ply_path = Path(ply_path)
        if not ply_path.exists():
            raise FileNotFoundError(f"Point cloud file not found: {ply_path}")
        self.pcd = o3d.io.read_point_cloud(str(ply_path))
        print(f"Loaded point cloud with {len(self.pcd.points)} points from {ply_path}")

    def create_point_cloud_from_rgbd(self, scale_depth=1.0, truncate_depth=0.75, min_depth=0.35):
        color_path = self.captures_dir / "color.png"
        if not color_path.exists():
            raise FileNotFoundError(f"Color image not found: {color_path}")
        color_img = o3d.io.read_image(str(color_path))

        depth_path = self.captures_dir / "depth_meters.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"Depth data not found: {depth_path}")
        depth_data = np.load(depth_path)
        depth_data[depth_data <= min_depth] = 0.0

        intrinsic_path = self.captures_dir / "intrinsic_data.json"
        if not intrinsic_path.exists():
            raise FileNotFoundError(f"Intrinsic data not found: {intrinsic_path}")
        with open(intrinsic_path) as f:
            intrinsics = json.load(f)

        color_np = np.asarray(color_img)
        h, w = depth_data.shape
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["ppx"]
        cy = intrinsics["ppy"]
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_data.astype(np.float64) * scale_depth
        z[(z <= 0) | (z > truncate_depth)] = 0.0
        mask = z > 0
        x = ((u - cx) * z / fx)[mask]
        y = ((v - cy) * z / fy)[mask]
        z_valid = z[mask]
        points = np.stack([x, y, z_valid], axis=1)
        colors = color_np[mask].astype(np.float64) / 255.0
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(points)
        self.pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"Point cloud created with {len(self.pcd.points)} points")

    def save_to_ply(self):
        o3d.io.write_point_cloud(str(self.ply_path), self.pcd)
        print(f"Saved point cloud to {self.ply_path}")

    def segment_plane(self, distance_threshold=0.02, ransac_n=3, num_iterations=10000):
        self.plane_model, inliers = self.pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=ransac_n,
            num_iterations=num_iterations,
        )
        self.pcd = self.pcd.select_by_index(inliers, invert=True)

    def crop_above_plane(self, max_height=0.20):
        a, b, c, d = self.plane_model
        norm = np.sqrt(a**2 + b**2 + c**2)
        points = np.asarray(self.pcd.points)
        dist = (a * points[:, 0] + b * points[:, 1] + c * points[:, 2] + d) / norm
        self.pcd = self.pcd.select_by_index(np.where(np.abs(dist) < max_height)[0])

    def crop_sides(self, x_range=(-0.15, 0.15), y_range=(-0.10, 0.10)):
        points = np.asarray(self.pcd.points)
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
        )
        self.pcd = self.pcd.select_by_index(np.where(mask)[0])

    def dbscan_objects(self, min_points_per_object=2000, colorize=False):
        labels = np.array(self.pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))
        if labels.size == 0 or labels.max() < 0:
            return []
        max_label = labels.max()
        if colorize:
            colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
            colors[labels < 0] = 0
            self.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        points = np.asarray(self.pcd.points)
        objects = []
        for i in range(max_label + 1):
            cluster_points = points[labels == i]
            if len(cluster_points) >= min_points_per_object:
                objects.append(
                    {
                        "label": i,
                        "centroid": cluster_points.mean(axis=0),
                        "points": cluster_points,
                        "num_points": len(cluster_points),
                    }
                )
        return objects

    def visualize(self, window_name="Point Cloud Visualization"):
        o3d.visualization.draw_geometries([self.pcd], window_name=window_name, width=1024, height=768)
