import numpy as np

from cube_vision.ik import mjcf_to_motor, traj_to_goals


def test_mjcf_to_motor_maps_expected_axes():
    q_deg = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    mapped = mjcf_to_motor(q_deg)
    assert np.allclose(mapped, np.array([-10.0, 70.0, -60.0, 40.0, 50.0]))


def test_traj_to_goals_preserves_joint_order():
    traj = [np.deg2rad(np.array([0.0, 90.0, 90.0, 5.0, 0.0]))]
    goals = traj_to_goals(traj, ["a", "b", "c", "d", "e"])
    assert goals == [{"a": -0.0, "b": 0.0, "c": 0.0, "d": 5.0, "e": 0.0}]
