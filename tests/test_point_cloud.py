"""Tests for cube_vision.vision.PointCloud."""

import json
import tempfile
from pathlib import Path

import numpy as np

from cube_vision.vision import PointCloud


def _create_fake_captures(tmpdir: Path, h: int = 10, w: int = 10):
    """Create minimal fake capture files for testing."""
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[3:7, 3:7] = (128, 64, 200)

    depth = np.full((h, w), 0.5, dtype=np.float32)
    depth[0, 0] = 0.0  # invalid pixel

    intrinsics = {"fx": 300.0, "fy": 300.0, "ppx": w / 2, "ppy": h / 2}

    import cv2
    cv2.imwrite(str(tmpdir / "color.png"), color)
    np.save(tmpdir / "depth_meters.npy", depth)
    with open(tmpdir / "intrinsic_data.json", "w") as f:
        json.dump(intrinsics, f)


def test_create_point_cloud_from_rgbd():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _create_fake_captures(tmpdir)
        pc = PointCloud(captures_dir=tmpdir)
        pc.create_point_cloud_from_rgbd(truncate_depth=1.0, min_depth=0.1)
        assert pc.pcd is not None
        assert len(pc.pcd.points) > 0


def test_save_and_load_ply():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _create_fake_captures(tmpdir)
        pc = PointCloud(captures_dir=tmpdir)
        pc.create_point_cloud_from_rgbd(truncate_depth=1.0, min_depth=0.1)
        pc.save_to_ply()
        assert pc.ply_path.exists()

        pc2 = PointCloud(captures_dir=tmpdir)
        pc2.load_from_ply(pc.ply_path)
        assert len(pc2.pcd.points) == len(pc.pcd.points)
