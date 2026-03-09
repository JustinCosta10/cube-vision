"""Motor configuration, calibration, and RealSense capture."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from cube_vision import CALIBRATION_DIR, REALSENSE_OUTPUT_DIR, load_config

# ---------------------------------------------------------------------------
# Build motor definitions from config.yaml
# ---------------------------------------------------------------------------

_NORM_MODES = {
    "degrees": MotorNormMode.DEGREES,
    "range_0_100": MotorNormMode.RANGE_0_100,
}


def _build_motor_defs(motor_cfg: dict) -> dict[str, Motor]:
    defs = {}
    for name, m in motor_cfg.items():
        mode = _NORM_MODES[m["mode"]]
        defs[name] = Motor(m["id"], m["model"], mode)
    return defs


def _load_hardware_config():
    """Read hardware settings from config.yaml once."""
    cfg = load_config()

    arm_bus_port = cfg["buses"]["arm"]["port"]
    head_bus_port = cfg["buses"]["head"]["port"]
    arm_motor_defs = _build_motor_defs(cfg["motors"]["arm"])
    head_motor_defs = _build_motor_defs(cfg["motors"]["head"])

    ctrl = cfg["control"]
    profile = PositionControlProfile(
        torque_limit=ctrl["torque_limit"],
        acceleration=ctrl["acceleration"],
        p=ctrl["pid"]["p"],
        i=ctrl["pid"]["i"],
        d=ctrl["pid"]["d"],
    )

    cam = cfg["camera"]
    det = cfg["detection"]

    return {
        "arm_bus_port": arm_bus_port,
        "head_bus_port": head_bus_port,
        "arm_motor_defs": arm_motor_defs,
        "head_motor_defs": head_motor_defs,
        "default_profile": profile,
        "camera_width": cam["width"],
        "camera_height": cam["height"],
        "camera_fps": cam["fps"],
        "default_detect_color": det["default_color"],
        "default_detect_exclude_bottom_fraction": det["exclude_bottom_fraction"],
        "default_ik_target_offset_m": tuple(cfg["ik"]["target_offset"]),
    }


@dataclass(frozen=True)
class PositionControlProfile:
    torque_limit: int = 500
    acceleration: int = 10
    p: int = 8
    i: int = 0
    d: int = 32


# Load once at import time.
_hw = _load_hardware_config()

# From config.yaml
ARM_BUS_PORT: str = _hw["arm_bus_port"]
HEAD_BUS_PORT: str = _hw["head_bus_port"]
ARM_MOTOR_DEFS: dict[str, Motor] = _hw["arm_motor_defs"]
HEAD_MOTOR_DEFS: dict[str, Motor] = _hw["head_motor_defs"]
DEFAULT_PROFILE: PositionControlProfile = _hw["default_profile"]
CAMERA_WIDTH: int = _hw["camera_width"]
CAMERA_HEIGHT: int = _hw["camera_height"]
CAMERA_FPS: int = _hw["camera_fps"]
DEFAULT_DETECT_COLOR: str = _hw["default_detect_color"]
DEFAULT_DETECT_EXCLUDE_BOTTOM_FRACTION: float = _hw["default_detect_exclude_bottom_fraction"]
DEFAULT_IK_TARGET_OFFSET_M: tuple = _hw["default_ik_target_offset_m"]

# Hardcoded (coupled to IK solver and MJCF model)
LEFT_ARM_JOINT_KEYS = [
    "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
]
RIGHT_ARM_JOINT_KEYS = [
    "shoulder_pan_2", "shoulder_lift_2", "elbow_flex_2", "wrist_flex_2", "wrist_roll_2",
]
LEFT_GRIPPER = "gripper"
RIGHT_GRIPPER = "gripper_2"
ALL_ARM_MOTORS = LEFT_ARM_JOINT_KEYS + [LEFT_GRIPPER] + RIGHT_ARM_JOINT_KEYS + [RIGHT_GRIPPER]

DEFAULT_ARM_CALIBRATION_FILE = CALIBRATION_DIR / "arm_bus.json"
DEFAULT_HEAD_CALIBRATION_FILE = CALIBRATION_DIR / "head_bus.json"


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def load_calibration(bus: FeetechMotorsBus, filepath: Path) -> dict:
    with open(filepath) as f:
        calib_raw = json.load(f)
    bus.calibration = {
        name: MotorCalibration(**vals) for name, vals in calib_raw.items()
    }
    print(f"Loaded calibration from {filepath}")
    return calib_raw


def run_interactive_calibration(bus: FeetechMotorsBus, filepath: Path) -> dict:
    motor_names = list(bus.motors.keys())
    bus.disable_torque(motor_names)

    input("\n>>> Move ALL motors to the MIDDLE of their range of motion, then press ENTER...")
    homing_offsets = bus.set_half_turn_homings(motor_names)
    print(f"Homing offsets set: {homing_offsets}")

    print("\n>>> Move ALL motors through their FULL range of motion.")
    input("    Move each joint to both extremes. Press ENTER when done...")
    range_mins, range_maxes = bus.record_ranges_of_motion(motor_names)
    print(f"Range mins: {range_mins}")
    print(f"Range maxes: {range_maxes}")

    calib_raw = {}
    for name in motor_names:
        motor = bus.motors[name]
        calib_raw[name] = {
            "id": motor.id,
            "drive_mode": 0,
            "homing_offset": homing_offsets[name],
            "range_min": range_mins[name],
            "range_max": range_maxes[name],
        }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(calib_raw, f, indent=4)
    print(f"Calibration saved to {filepath}")

    bus.calibration = {
        name: MotorCalibration(**vals) for name, vals in calib_raw.items()
    }
    return calib_raw


def load_or_run_calibration(
    bus: FeetechMotorsBus,
    filepath: Path = DEFAULT_ARM_CALIBRATION_FILE,
    force: bool = False,
) -> dict:
    if filepath.exists() and not force:
        return load_calibration(bus, filepath)
    if force:
        print("Force recalibration requested.")
    else:
        print("No calibration file found. Running calibration...")
    return run_interactive_calibration(bus, filepath)


# ---------------------------------------------------------------------------
# Motor connection and control
# ---------------------------------------------------------------------------


def connect_arm_bus(calibrate: bool = True) -> FeetechMotorsBus:
    bus = FeetechMotorsBus(port=ARM_BUS_PORT, motors=ARM_MOTOR_DEFS)
    bus.connect()
    if calibrate:
        load_or_run_calibration(bus, filepath=DEFAULT_ARM_CALIBRATION_FILE)
    return bus


def connect_head_bus(calibrate: bool = True) -> FeetechMotorsBus:
    bus = FeetechMotorsBus(port=HEAD_BUS_PORT, motors=HEAD_MOTOR_DEFS)
    bus.connect()
    if calibrate:
        load_or_run_calibration(bus, filepath=DEFAULT_HEAD_CALIBRATION_FILE)
    return bus


def apply_position_control_profile(
    bus: FeetechMotorsBus,
    motors: list[str],
    profile: PositionControlProfile,
) -> None:
    print("Applying motor limits:")
    print(f"    Torque_Limit = {profile.torque_limit} / 1000")
    print(f"    Acceleration = {profile.acceleration} / 254")
    print(f"    P={profile.p}, I={profile.i}, D={profile.d}\n")

    bus.disable_torque(motors)
    for name in motors:
        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
        bus.write("Torque_Limit", name, profile.torque_limit)
        bus.write("Acceleration", name, profile.acceleration)
        bus.write("P_Coefficient", name, profile.p)
        bus.write("I_Coefficient", name, profile.i)
        bus.write("D_Coefficient", name, profile.d)
    bus.enable_torque(motors)


def apply_default_arm_profile(bus: FeetechMotorsBus) -> None:
    apply_position_control_profile(bus, ALL_ARM_MOTORS, DEFAULT_PROFILE)


def read_head_positions_deg(bus: FeetechMotorsBus) -> tuple[float, float]:
    head_pos = bus.sync_read("Present_Position", ["head_pan", "head_tilt"])
    return float(head_pos["head_pan"]), float(head_pos["head_tilt"])


# ---------------------------------------------------------------------------
# RealSense capture
# ---------------------------------------------------------------------------


def capture(out_dir: Path | None = None) -> Path:
    out_dir = REALSENSE_OUTPUT_DIR if out_dir is None else Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, CAMERA_FPS)
    config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, CAMERA_FPS)

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


# ---------------------------------------------------------------------------
# Calibration CLI
# ---------------------------------------------------------------------------


def calibrate_cli() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SO-101 motors")
    parser.add_argument("--bus", choices=["arm", "head", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    buses = []
    try:
        if args.bus in ("arm", "all"):
            arm_bus = FeetechMotorsBus(port=ARM_BUS_PORT, motors=ARM_MOTOR_DEFS)
            arm_bus.connect()
            buses.append(("arm", arm_bus, DEFAULT_ARM_CALIBRATION_FILE))

        if args.bus in ("head", "all"):
            head_bus = FeetechMotorsBus(port=HEAD_BUS_PORT, motors=HEAD_MOTOR_DEFS)
            head_bus.connect()
            buses.append(("head", head_bus, DEFAULT_HEAD_CALIBRATION_FILE))

        for label, bus, calib_file in buses:
            print(f"\n=== Calibrating {label} bus ===")
            load_or_run_calibration(bus, filepath=calib_file, force=args.force)
            print(f"\nCalibrated positions ({label}):")
            positions = bus.sync_read("Present_Position", list(bus.motors.keys()))
            for name, val in positions.items():
                print(f"  {name}: {float(val):.2f}")
    finally:
        for _, bus, _ in buses:
            bus.disconnect()


if __name__ == "__main__":
    calibrate_cli()
