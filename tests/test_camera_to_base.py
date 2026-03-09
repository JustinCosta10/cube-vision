"""Tests for cube_vision.transforms."""

import numpy as np

from cube_vision.transforms import (
    _head_motor_to_mjcf,
    camera_xyz_to_base_xyz,
    camera_xyz_to_base2_xyz,
    head_fk,
)


def test_head_motor_to_mjcf_subtracts_tilt_offset():
    q_deg = np.array([10.0, 30.0])
    result = _head_motor_to_mjcf(q_deg)
    assert result[0] == 10.0
    assert result[1] == 16.0  # 30 - 14


def test_head_motor_to_mjcf_does_not_mutate_input():
    q_deg = np.array([5.0, 20.0])
    _head_motor_to_mjcf(q_deg)
    assert q_deg[1] == 20.0


def test_head_fk_returns_4x4():
    T = head_fk(np.array([0.0, 0.0]))
    assert T.shape == (4, 4)
    assert T[3, 3] == 1.0
    assert np.allclose(T[3, :3], 0.0)


def test_head_fk_rotation_is_orthonormal():
    T = head_fk(np.array([0.0, 0.0]))
    R = T[:3, :3]
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_camera_xyz_identity_returns_three_floats():
    joint_values = {"head_pan_joint": 0.0, "head_tilt_joint": 0.0}
    result = camera_xyz_to_base_xyz(0.0, 0.0, 0.3, joint_values)
    assert len(result) == 3
    assert all(isinstance(v, float) for v in result)


def test_base_and_base2_differ():
    joint_values = {"head_pan_joint": 0.0, "head_tilt_joint": 0.0}
    b1 = camera_xyz_to_base_xyz(0.0, 0.05, 0.3, joint_values)
    b2 = camera_xyz_to_base2_xyz(0.0, 0.05, 0.3, joint_values)
    # The two base frames are at different positions, so results should differ
    assert not np.allclose(b1, b2, atol=1e-6)
