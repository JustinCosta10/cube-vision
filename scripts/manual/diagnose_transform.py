#!/usr/bin/env python3
"""Diagnostic: compare FK end-effector position against vision result."""

import numpy as np
import pinocchio as pin

from cube_vision import DEG2RAD, MJCF_PATH
from cube_vision.hardware import (
    capture,
    connect_arm_bus,
    connect_head_bus,
    load_or_run_calibration,
    read_head_positions_deg,
)
from cube_vision.vision import detect_object
from cube_vision.transforms import _head_motor_to_mjcf, camera_xyz_to_base_xyz


def motor_to_mjcf(q_deg):
    """Inverse of mjcf_to_motor: motor degrees -> MJCF degrees."""
    out = q_deg.copy()
    out[0] = -out[0]
    out[1] = 90.0 - out[1]
    out[2] = out[2] + 90.0
    return out


def main() -> None:
    arm_bus = connect_arm_bus()
    head_bus = connect_head_bus()
    try:
        all_arm_motors = list(arm_bus.motors.keys())
        arm_bus.disable_torque(all_arm_motors)
        input("\n>>> Place the gripper tip ON the cube, then press ENTER...")

        positions = arm_bus.sync_read("Present_Position", all_arm_motors)
        head_pan_deg, head_tilt_deg = read_head_positions_deg(head_bus)
        arm_motor_deg = np.array(
            [
                float(positions["shoulder_pan"]),
                float(positions["shoulder_lift"]),
                float(positions["elbow_flex"]),
                float(positions["wrist_flex"]),
                float(positions["wrist_roll"]),
            ]
        )

        input("\n>>> Now move the gripper OUT OF THE WAY of the cube, then press ENTER...")

        full_model = pin.buildModelFromMJCF(str(MJCF_PATH))
        q_neutral = pin.neutral(full_model)
        arm_joints = {"Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"}
        lock_ids = [i for i in range(1, full_model.njoints) if full_model.names[i] not in arm_joints]
        arm_model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        arm_data = arm_model.createData()
        q = pin.neutral(arm_model)
        mjcf_rad = motor_to_mjcf(arm_motor_deg) * DEG2RAD
        for i, jname in enumerate(["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"]):
            jid = arm_model.getJointId(jname)
            q[arm_model.joints[jid].idx_q] = mjcf_rad[i]

        pin.forwardKinematics(arm_model, arm_data, q)
        pin.updateFramePlacements(arm_model, arm_data)
        ee_frame_id = arm_model.getFrameId("Gripper_Tip")
        base_frame_id = arm_model.getFrameId("Base")
        oMee = arm_data.oMf[ee_frame_id]
        oMbase = arm_data.oMf[base_frame_id]
        ee_in_base = oMbase.rotation.T @ (oMee.translation - oMbase.translation)

        capture()
        centroid_cam = detect_object(color="red")
        head_mjcf = _head_motor_to_mjcf(np.array([head_pan_deg, head_tilt_deg]))
        print(f"Head motor deg: {head_pan_deg:.2f}, {head_tilt_deg:.2f}")
        print(f"Head MJCF deg: {head_mjcf[0]:.2f}, {head_mjcf[1]:.2f}")

        vx, vy, vz = camera_xyz_to_base_xyz(
            centroid_cam[0],
            centroid_cam[1],
            centroid_cam[2],
            {
                "head_pan_joint": head_pan_deg * DEG2RAD,
                "head_tilt_joint": head_tilt_deg * DEG2RAD,
            },
        )
        err = np.array([vx - ee_in_base[0], vy - ee_in_base[1], vz - ee_in_base[2]])
        print(f"FK:  {ee_in_base}")
        print(f"VIS: {[vx, vy, vz]}")
        print(f"ERR: {err}, |err|={np.linalg.norm(err)*100:.1f} cm")
    finally:
        arm_bus.disconnect()
        head_bus.disconnect()


if __name__ == "__main__":
    main()
