import os
import json
from pathlib import Path

os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from cube_vision.config.paths import REALSENSE_OUTPUT_DIR


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
