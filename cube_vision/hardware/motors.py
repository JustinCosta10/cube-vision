from dataclasses import dataclass

from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from cube_vision.config.robot import (
    ALL_ARM_MOTORS,
    ARM_BUS_PORT,
    ARM_MOTOR_DEFS,
    DEFAULT_ARM_CALIBRATION_FILE,
    DEFAULT_HEAD_CALIBRATION_FILE,
    HEAD_BUS_PORT,
    HEAD_MOTOR_DEFS,
)
from cube_vision.hardware.calibration import load_or_run_calibration


@dataclass(frozen=True)
class PositionControlProfile:
    torque_limit: int = 500
    acceleration: int = 10
    p: int = 8
    i: int = 0
    d: int = 32


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
    apply_position_control_profile(bus, ALL_ARM_MOTORS, PositionControlProfile())


def read_head_positions_deg(bus: FeetechMotorsBus) -> tuple[float, float]:
    head_pos = bus.sync_read("Present_Position", ["head_pan", "head_tilt"])
    return float(head_pos["head_pan"]), float(head_pos["head_tilt"])
