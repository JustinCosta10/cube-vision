import json
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from cube_vision.config.paths import REALSENSE_OUTPUT_DIR


def capture(out_dir: Path | None = None) -> Path:
    out_dir = REALSENSE_OUTPUT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    profile = pipeline.start(config)
    try:
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()
        print("Depth scale:", depth_scale, "meters/unit")

        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipeline.wait_for_frames()

        frames = align.process(pipeline.wait_for_frames())
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("Failed to get depth or color frame")

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        cv2.imwrite(str(out_dir / "color.png"), color_image)
        cv2.imwrite(str(out_dir / "depth_16u.png"), depth_image)

        depth_m = depth_image.astype(np.float32) * depth_scale
        np.save(out_dir / "depth_meters.npy", depth_m)

        depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
        cv2.imwrite(str(out_dir / "depth_vis.png"), depth_vis.astype(np.uint8))

        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsic_data = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "model": str(intrinsics.model),
        }
        with open(out_dir / "intrinsic_data.json", "w") as f:
            json.dump(intrinsic_data, f, indent=4)
    finally:
        pipeline.stop()

    print("Saved to:", out_dir)
    return out_dir
