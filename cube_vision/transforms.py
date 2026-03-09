"""Camera-to-base frame coordinate transforms using Pinocchio FK."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pinocchio as pin

from cube_vision import MJCF_PATH

_model = pin.buildModelFromMJCF(str(MJCF_PATH))
_data = _model.createData()

_BASE_FRAME_ID = _model.getFrameId("Base")
_BASE_2_FRAME_ID = _model.getFrameId("Base_2")
_CAMERA_LINK_FRAME_ID = _model.getFrameId("head_camera_link")
_R_LINK_TO_OPTICAL = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
_HEAD_PAN_IDX = _model.joints[_model.getJointId("head_pan_joint")].idx_q
_HEAD_TILT_IDX = _model.joints[_model.getJointId("head_tilt_joint")].idx_q


def _head_motor_to_mjcf(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[1] = out[1] - 14.0
    return out


def head_fk(q_urdf_rad: np.ndarray) -> np.ndarray:
    q = pin.neutral(_model)
    q[_HEAD_PAN_IDX] = q_urdf_rad[0]
    q[_HEAD_TILT_IDX] = q_urdf_rad[1]
    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)

    oMf = _data.oMf[_CAMERA_LINK_FRAME_ID]
    T = np.eye(4)
    T[:3, :3] = oMf.rotation @ _R_LINK_TO_OPTICAL
    T[:3, 3] = oMf.translation
    return T


def _camera_xyz_to_frame_xyz(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
    base_frame_id: int,
) -> Tuple[float, float, float]:
    pan_motor_rad = joint_values.get("head_pan_joint", 0.0)
    tilt_motor_rad = joint_values.get("head_tilt_joint", 0.0)

    motor_deg = np.array([np.rad2deg(pan_motor_rad), np.rad2deg(tilt_motor_rad)])
    mjcf_deg = _head_motor_to_mjcf(motor_deg)

    q = pin.neutral(_model)
    q[_HEAD_PAN_IDX] = np.deg2rad(mjcf_deg[0])
    q[_HEAD_TILT_IDX] = np.deg2rad(mjcf_deg[1])
    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)

    oMbase = _data.oMf[base_frame_id]
    oMcam_link = _data.oMf[_CAMERA_LINK_FRAME_ID]
    R_cam_optical = oMcam_link.rotation @ _R_LINK_TO_OPTICAL

    T = np.eye(4)
    T[:3, :3] = oMbase.rotation.T @ R_cam_optical
    T[:3, 3] = oMbase.rotation.T @ (oMcam_link.translation - oMbase.translation)

    p_base = (T @ np.array([x, y, z, 1.0], dtype=float))[:3]
    return float(p_base[0]), float(p_base[1]), float(p_base[2])


def camera_xyz_to_base_xyz(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
) -> Tuple[float, float, float]:
    return _camera_xyz_to_frame_xyz(x, y, z, joint_values, _BASE_FRAME_ID)


def camera_xyz_to_base2_xyz(
    x: float,
    y: float,
    z: float,
    joint_values: Dict[str, float],
) -> Tuple[float, float, float]:
    return _camera_xyz_to_frame_xyz(x, y, z, joint_values, _BASE_2_FRAME_ID)
