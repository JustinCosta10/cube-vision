import numpy as np

from cube_vision.config.robot import RAD2DEG


def mjcf_to_motor(q_deg: np.ndarray) -> np.ndarray:
    out = q_deg.copy()
    out[0] = -out[0]
    out[1] = 90.0 - out[1]
    out[2] = out[2] - 90.0
    return out


def traj_to_goals(traj_rad: list[np.ndarray], joint_keys: list[str]) -> list[dict[str, float]]:
    stack = np.stack(traj_rad)
    traj_deg = np.array([mjcf_to_motor(q * RAD2DEG) for q in stack])
    return [
        {joint: float(q_deg[i]) for i, joint in enumerate(joint_keys)}
        for q_deg in traj_deg
    ]
