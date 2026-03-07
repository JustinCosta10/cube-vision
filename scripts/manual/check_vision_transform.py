#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from lerobot.motors.feetech import FeetechMotorsBus

from cube_vision.config.robot import DEG2RAD, HEAD_MOTOR_DEFS, HEAD_BUS_PORT
from cube_vision.hardware.calibration import load_or_run_calibration
from cube_vision.hardware.realsense import capture
from cube_vision.perception.color import detect_object
from cube_vision.transforms.camera_to_base import camera_xyz_to_base_xyz

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"


def run_test(color: str, ground_truth: np.ndarray | None, skip_capture: bool):
    bus = FeetechMotorsBus(port=HEAD_BUS_PORT, motors=HEAD_MOTOR_DEFS)
    bus.connect()
    try:
        load_or_run_calibration(bus, filepath=Path(__file__).resolve().parents[2] / "calibration" / "head_bus.json")
        head_pos = bus.sync_read("Present_Position", ["head_pan", "head_tilt"])
        head_pan_deg = float(head_pos["head_pan"])
        head_tilt_deg = float(head_pos["head_tilt"])

        if not skip_capture:
            capture()
        centroid_cam = detect_object(color=color)
        bx, by, bz = camera_xyz_to_base_xyz(
            centroid_cam[0],
            centroid_cam[1],
            centroid_cam[2],
            {
                "head_pan_joint": head_pan_deg * DEG2RAD,
                "head_tilt_joint": head_tilt_deg * DEG2RAD,
            },
        )
        computed = np.array([bx, by, bz])
        if ground_truth is None:
            raw = input("x y z: ").strip().split()
            if len(raw) != 3:
                return
            ground_truth = np.array([float(v) for v in raw])

        error = computed - ground_truth
        dist = np.linalg.norm(error)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        report_path = OUTPUT_DIR / "vision_transform_test.txt"
        with open(report_path, "w") as f:
            f.write(f"computed_base: {computed.tolist()}\n")
            f.write(f"ground_truth_base: {ground_truth.tolist()}\n")
            f.write(f"error_per_axis: {error.tolist()}\n")
            f.write(f"euclidean_error_m: {dist:.6f}\n")
        print(f"Computed: {computed}")
        print(f"Ground truth: {ground_truth}")
        print(f"Error: {error}, dist={dist:.4f} m")
    finally:
        bus.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual vision transform check")
    parser.add_argument("--color", default="red", choices=["red", "green", "blue"])
    parser.add_argument("--truth", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--skip-capture", action="store_true")
    args = parser.parse_args()
    run_test(args.color, np.array(args.truth) if args.truth else None, args.skip_capture)
