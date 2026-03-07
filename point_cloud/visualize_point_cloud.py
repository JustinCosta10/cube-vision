import open3d as o3d

from cube_vision.config.paths import REALSENSE_OUTPUT_DIR


if __name__ == "__main__":
    ply_path = REALSENSE_OUTPUT_DIR / "vision.ply"
    print(f"Loading point cloud from: {ply_path}")
    pcd = o3d.io.read_point_cloud(str(ply_path))
    print(f"Point cloud has {len(pcd.points)} points")
    print(f"Point cloud has colors: {pcd.has_colors()}")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Point Cloud Visualization",
        width=1024,
        height=768,
        point_show_normal=False,
    )
