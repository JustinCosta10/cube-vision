"""Tests for cube_vision.ik."""

import numpy as np

from cube_vision.ik import IK_SO101


def test_ik_so101_init():
    ik = IK_SO101()
    assert ik.model is not None
    assert ik.data is not None
    assert ik.EE_LEFT == "Fixed_Jaw"
    assert ik.EE_RIGHT == "Fixed_Jaw_2"


def test_base_to_world_and_back():
    ik = IK_SO101()
    p_base = np.array([0.0, -0.2, 0.05])
    p_world = ik.base_to_world(p_base)
    # Inverse: p_base = R^T @ (p_world - t)
    p_back = ik._base_R.T @ (p_world - ik._base_t)
    assert np.allclose(p_base, p_back, atol=1e-10)


def test_choose_arm_prefers_closer():
    ik = IK_SO101()
    left_close = np.array([0.0, -0.1, 0.0])
    right_far = np.array([0.5, -0.5, 0.0])
    assert ik.choose_arm(left_close, right_far) == "left"
    assert ik.choose_arm(right_far, left_close) == "right"


def test_generate_ik_reachable_target():
    ik = IK_SO101()
    target = [0.0, -0.20, 0.05]
    traj = ik.generate_ik(target_xyz=target, gripper_offset_xyz=[0.0, 0.0, 0.0])
    assert len(traj) > 0
    # Each waypoint should have 5 joint values (left arm)
    assert traj[-1].shape == (5,)


def test_generate_ik_unreachable_target_returns_empty():
    ik = IK_SO101()
    target = [10.0, 10.0, 10.0]  # way out of reach
    traj = ik.generate_ik(target_xyz=target, gripper_offset_xyz=[0.0, 0.0, 0.0])
    assert traj == []
