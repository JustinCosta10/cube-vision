#!/usr/bin/env python3
"""Compatibility wrapper for calibration CLI."""

from cube_vision.config.robot import (
    ARM_BUS_PORT,
    ARM_MOTOR_DEFS,
    DEFAULT_ARM_CALIBRATION_FILE,
    DEFAULT_HEAD_CALIBRATION_FILE,
    HEAD_BUS_PORT,
    HEAD_MOTOR_DEFS,
    LEGACY_BUS_PORT as BUS_PORT,
    LEGACY_MOTOR_DEFS as MOTOR_DEFS,
)
from cube_vision.hardware.calibration import (
    load_calibration,
    load_or_run_calibration,
    main,
    run_interactive_calibration,
)

__all__ = [
    "ARM_BUS_PORT",
    "ARM_MOTOR_DEFS",
    "BUS_PORT",
    "DEFAULT_ARM_CALIBRATION_FILE",
    "DEFAULT_HEAD_CALIBRATION_FILE",
    "HEAD_BUS_PORT",
    "HEAD_MOTOR_DEFS",
    "MOTOR_DEFS",
    "load_calibration",
    "load_or_run_calibration",
    "run_interactive_calibration",
]

if __name__ == "__main__":
    main()
