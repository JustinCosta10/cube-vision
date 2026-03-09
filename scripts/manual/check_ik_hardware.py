"""Manual hardware check for bimanual IK."""

import time

import numpy as np

from cube_vision.hardware import LEFT_ARM_JOINT_KEYS, RIGHT_ARM_JOINT_KEYS, apply_default_arm_profile, connect_arm_bus
from cube_vision.ik import IK_SO101, traj_to_goals

TARGET_BASE = [0.0, -0.20, 0.00]
ARM = "auto"
IK_TARGET_OFFSET_X_M = -0.065
IK_TARGET_OFFSET_Y_M = 0.0
IK_TARGET_OFFSET_Z_M = 0.0


def main() -> None:
    arm_bus = connect_arm_bus()
    try:
        apply_default_arm_profile(arm_bus)
        ik = IK_SO101()
        offset = np.array([IK_TARGET_OFFSET_X_M, IK_TARGET_OFFSET_Y_M, IK_TARGET_OFFSET_Z_M])
        target_left = np.array(TARGET_BASE) + offset
        target_world = ik.base_to_world(target_left)
        target_right = ik._base2_R.T @ (target_world - ik._base2_t)

        if ARM == "auto":
            chosen = ik.choose_arm(target_left, target_right)
        elif ARM in ("left", "right"):
            chosen = ARM
        else:
            raise ValueError(f"ARM must be 'left', 'right', or 'auto', got '{ARM}'")

        active_target = target_left if chosen == "left" else target_right
        joint_keys = LEFT_ARM_JOINT_KEYS if chosen == "left" else RIGHT_ARM_JOINT_KEYS
        gripper = "gripper" if chosen == "left" else "gripper_2"
        traj = ik.generate_ik_bimanual(active_target.tolist(), arm=chosen)
        if not traj:
            raise SystemExit("IK failed; target may be out of reach.")

        goals = traj_to_goals(traj, joint_keys)
        for goal in goals:
            goal[gripper] = 100.0
            arm_bus.sync_write("Goal_Position", goal)
            time.sleep(0.01)
    finally:
        arm_bus.disconnect()


if __name__ == "__main__":
    main()
