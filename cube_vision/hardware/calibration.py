#!/usr/bin/env python3
"""Motor calibration helpers and CLI."""

import argparse
import json
from pathlib import Path

from lerobot.motors import MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus

from cube_vision.config.robot import (
    ARM_BUS_PORT,
    ARM_MOTOR_DEFS,
    DEFAULT_ARM_CALIBRATION_FILE,
    DEFAULT_HEAD_CALIBRATION_FILE,
    HEAD_BUS_PORT,
    HEAD_MOTOR_DEFS,
)


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


def main() -> None:
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
    main()
