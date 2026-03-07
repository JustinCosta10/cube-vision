from pathlib import Path

from lerobot.motors import Motor, MotorNormMode

from cube_vision.config.paths import CALIBRATION_DIR

DEG2RAD = 3.141592653589793 / 180.0
RAD2DEG = 180.0 / 3.141592653589793

ARM_BUS_PORT = "/dev/ttyACM0"
HEAD_BUS_PORT = "/dev/ttyACM1"

DEFAULT_ARM_CALIBRATION_FILE = CALIBRATION_DIR / "arm_bus.json"
DEFAULT_HEAD_CALIBRATION_FILE = CALIBRATION_DIR / "head_bus.json"

ARM_MOTOR_DEFS = {
    "shoulder_pan_2": Motor(1, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift_2": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex_2": Motor(3, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex_2": Motor(4, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll_2": Motor(5, "sts3215", MotorNormMode.DEGREES),
    "gripper_2": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    "shoulder_pan": Motor(7, "sts3215", MotorNormMode.DEGREES),
    "shoulder_lift": Motor(8, "sts3215", MotorNormMode.DEGREES),
    "elbow_flex": Motor(9, "sts3215", MotorNormMode.DEGREES),
    "wrist_flex": Motor(10, "sts3215", MotorNormMode.DEGREES),
    "wrist_roll": Motor(11, "sts3215", MotorNormMode.DEGREES),
    "gripper": Motor(12, "sts3215", MotorNormMode.RANGE_0_100),
}

HEAD_MOTOR_DEFS = {
    "head_pan": Motor(2, "sts3215", MotorNormMode.DEGREES),
    "head_tilt": Motor(1, "sts3215", MotorNormMode.DEGREES),
}

LEGACY_BUS_PORT = ARM_BUS_PORT
LEGACY_MOTOR_DEFS = {**ARM_MOTOR_DEFS, **HEAD_MOTOR_DEFS}

LEFT_ARM_JOINT_KEYS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]
RIGHT_ARM_JOINT_KEYS = [
    "shoulder_pan_2",
    "shoulder_lift_2",
    "elbow_flex_2",
    "wrist_flex_2",
    "wrist_roll_2",
]
ALL_ARM_MOTORS = LEFT_ARM_JOINT_KEYS + ["gripper"] + RIGHT_ARM_JOINT_KEYS + ["gripper_2"]

DEFAULT_IK_TARGET_OFFSET_M = (0.0, 0.0, 0.0)
DEFAULT_DETECT_EXCLUDE_BOTTOM_FRACTION = 0.10
