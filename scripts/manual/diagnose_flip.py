#!/usr/bin/env python3
"""Diagnostic dump of camera-to-base orientation at neutral head pose."""

import numpy as np
import pinocchio as pin

from cube_vision import MJCF_PATH

np.set_printoptions(precision=4, suppress=True)


def main() -> None:
    model = pin.buildModelFromMJCF(str(MJCF_PATH))
    data = model.createData()
    base_id = model.getFrameId("Base")
    cam_link_id = model.getFrameId("head_camera_link")
    r_link_to_optical = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])

    q = pin.neutral(model)
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    oMbase = data.oMf[base_id]
    oMcam = data.oMf[cam_link_id]
    r_cam_optical = oMcam.rotation @ r_link_to_optical
    r = oMbase.rotation.T @ r_cam_optical
    t = oMbase.rotation.T @ (oMcam.translation - oMbase.translation)

    print("Base translation:", oMbase.translation)
    print("Camera translation:", oMcam.translation)
    print("R_base_from_optical:\n", r)
    print("t:", t)

    test_points = {
        "center, 50cm forward": [0.0, 0.0, 0.5],
        "10cm RIGHT, 50cm fwd": [0.1, 0.0, 0.5],
        "10cm LEFT, 50cm fwd": [-0.1, 0.0, 0.5],
        "10cm DOWN, 50cm fwd": [0.0, 0.1, 0.5],
        "10cm UP, 50cm fwd": [0.0, -0.1, 0.5],
    }
    for label, p_opt in test_points.items():
        p_base = r @ np.array(p_opt) + t
        print(label, "->", p_base)


if __name__ == "__main__":
    main()
