import time

import numpy as np

from cube_vision.config.robot import (
    ALL_ARM_MOTORS,
    DEFAULT_DETECT_EXCLUDE_BOTTOM_FRACTION,
    DEFAULT_IK_TARGET_OFFSET_M,
    DEG2RAD,
    LEFT_ARM_JOINT_KEYS,
    RIGHT_ARM_JOINT_KEYS,
)
from cube_vision.hardware.motors import (
    apply_default_arm_profile,
    connect_arm_bus,
    connect_head_bus,
    read_head_positions_deg,
)
from cube_vision.hardware.realsense import capture
from cube_vision.perception.color import detect_object
from cube_vision.planning.ik_so101 import IK_SO101
from cube_vision.planning.joint_mapping import traj_to_goals
from cube_vision.transforms.camera_to_base import camera_xyz_to_base2_xyz, camera_xyz_to_base_xyz
from cube_vision.visualization.color_detection import visualize as visualize_color_detect
from cube_vision.visualization.ik import save_ik_plot


def run_pick_by_color(color: str = "red") -> None:
    arm_bus = connect_arm_bus()
    head_bus = connect_head_bus()
    try:
        head_pan_deg, head_tilt_deg = read_head_positions_deg(head_bus)
        print(f"Head motors (deg): pan={head_pan_deg:.2f}, tilt={head_tilt_deg:.2f}")

        apply_default_arm_profile(arm_bus)
        capture()

        centroid = detect_object(
            color=color,
            exclude_bottom_fraction=DEFAULT_DETECT_EXCLUDE_BOTTOM_FRACTION,
        )
        print(f"Camera centroid (optical frame): {centroid}")
        visualize_color_detect(
            color=color,
            head_pan_deg=head_pan_deg,
            head_tilt_deg=head_tilt_deg,
            out_name=f"color_detect_vis_grab_{time.strftime('%Y%m%d_%H%M%S')}.png",
            show_window=True,
            window_ms=0,
            exclude_bottom_fraction=DEFAULT_DETECT_EXCLUDE_BOTTOM_FRACTION,
        )

        joint_values = {
            "head_pan_joint": head_pan_deg * DEG2RAD,
            "head_tilt_joint": head_tilt_deg * DEG2RAD,
        }
        target_base = np.array(camera_xyz_to_base_xyz(centroid[0], centroid[1], centroid[2], joint_values))
        target_base2 = np.array(camera_xyz_to_base2_xyz(centroid[0], centroid[1], centroid[2], joint_values))

        offset = np.array(DEFAULT_IK_TARGET_OFFSET_M)
        target_base = target_base + offset
        target_base2 = target_base2 + offset
        print(f"Transformed to Base frame (left arm):  {target_base}")
        print(f"Transformed to Base_2 frame (right arm): {target_base2}")

        ik_solve = IK_SO101()
        chosen_arm = ik_solve.choose_arm(target_base, target_base2)
        if chosen_arm == "left":
            active_target = target_base
            active_joint_keys = LEFT_ARM_JOINT_KEYS
            active_gripper = "gripper"
            base_pos = ik_solve._base_t
        else:
            active_target = target_base2
            active_joint_keys = RIGHT_ARM_JOINT_KEYS
            active_gripper = "gripper_2"
            base_pos = ik_solve._base2_t

        print(f"IK target ({chosen_arm} arm): {active_target}")
        save_ik_plot(base_pos=base_pos, ik_target_base=np.array(active_target), camera_centroid_cam=np.array(centroid))

        traj = ik_solve.generate_ik_bimanual(active_target.tolist(), arm=chosen_arm)
        if not traj:
            raise RuntimeError("IK failed for target")

        goals = traj_to_goals(traj, active_joint_keys)
        print(f"Sending {len(goals)} waypoints...")
        for goal in goals:
            goal[active_gripper] = 100.0
            arm_bus.sync_write("Goal_Position", goal)
            time.sleep(0.01)
    finally:
        arm_bus.disconnect()
        head_bus.disconnect()
