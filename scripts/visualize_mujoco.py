#!/usr/bin/env python3
"""MuJoCo visualizer — simulates the full pick-by-color pipeline.

Spawns a cube, picks the best arm, solves IK, and approaches.
No physics — just kinematic animation.

Usage:
    python visualize_mujoco.py                     # random cube positions, auto arm selection
    python visualize_mujoco.py --cube-x 0.05 --cube-y -0.25 --cube-z 0.02
    python visualize_mujoco.py --speed 2.0         # faster playback
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# On macOS, MuJoCo's launch_passive requires mjpython. Auto-relaunch.
# ---------------------------------------------------------------------------
if platform.system() == "Darwin" and "MJPYTHON" not in os.environ:
    mjpython = shutil.which("mjpython")
    if mjpython:
        env = os.environ.copy()
        env["MJPYTHON"] = "1"
        result = subprocess.run([mjpython] + sys.argv, env=env)
        sys.exit(result.returncode)
    else:
        print("ERROR: mjpython not found. On macOS, MuJoCo viewer requires mjpython.")
        print("       It should be installed with mujoco: pip install mujoco")
        sys.exit(1)

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: mujoco package not found. Install with: pip install mujoco")
    sys.exit(1)

from cube_vision.ik import IK_SO101
from cube_vision.transforms import camera_xyz_to_base_xyz, camera_xyz_to_base2_xyz

_MJCF_PATH = _REPO_ROOT / "model" / "xlerobot.xml"

_LEFT_JOINT_NAMES = [
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
]
_RIGHT_JOINT_NAMES = [
    "Rotation_R",
    "Pitch_R",
    "Elbow_R",
    "Wrist_Pitch_R",
    "Wrist_Roll_R",
]
_LEFT_JAW = "Jaw_L"
_RIGHT_JAW = "Jaw_R"
_LEFT_EE_BODY = "Gripper_Tip"
_RIGHT_EE_BODY = "Gripper_Tip_2"

# Jaw joint limits (from MJCF): open ~ -0.37, closed ~ 1.74
_JAW_OPEN = -0.37
_JAW_CLOSED = 1.2  # don't fully max out

# Home pose: arms curled up with the gripper rotated horizontal
# [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll] in radians.
_HOME_POSE_RAD = np.deg2rad([0.0, 180.0, 180.0, 0.0, 90.0])


# Table surface height in MJCF world coords.
# Arm bases are at ~z=0.79 in MJCF. Table just below the base frames.
_TABLE_Z = 0.75
_TABLE_HALF_SIZE = (0.18, 0.40, 0.02)  # x, y, z half-extents
# Table center in MJCF world: in front of the robot (negative x), offset to avoid cart collision
_TABLE_POS = (-0.36, 0.0, _TABLE_Z - _TABLE_HALF_SIZE[2])
_CUBE_HALF = 0.017  # half-size of the cube geom


# ---------------------------------------------------------------------------
# MJCF helpers
# ---------------------------------------------------------------------------


def _inject_scene(root: ET.Element) -> None:
    """Inject ground plane, table, and target cube into the MJCF worldbody."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    existing = {b.get("name") for b in worldbody.findall("body")}
    existing |= {g.get("name") for g in worldbody.findall("geom")}

    # Ground plane
    if "ground" not in existing:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "ground",
                "type": "plane",
                "size": "2 2 0.01",
                "pos": "0 0 0",
                "rgba": "0.85 0.85 0.80 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    # Table
    if "table" not in existing:
        tx, ty, tz = _TABLE_POS
        sx, sy, sz = _TABLE_HALF_SIZE
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "table",
                "type": "box",
                "size": f"{sx} {sy} {sz}",
                "pos": f"{tx} {ty} {tz}",
                "rgba": "0.45 0.30 0.18 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    # Target cube (mocap so we can move it freely)
    if "target_cube" not in existing:
        cube_start_z = _TABLE_Z + _CUBE_HALF
        body = ET.SubElement(
            worldbody,
            "body",
            {"name": "target_cube", "mocap": "true", "pos": f"0 0 {cube_start_z}"},
        )
        ET.SubElement(
            body,
            "geom",
            {
                "name": "target_cube_geom",
                "type": "box",
                "size": f"{_CUBE_HALF} {_CUBE_HALF} {_CUBE_HALF}",
                "rgba": "1 0.1 0.1 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )

    # End-effector marker — bright sphere showing Pinocchio FK position
    if "ee_marker" not in existing:
        body = ET.SubElement(
            worldbody,
            "body",
            {"name": "ee_marker", "mocap": "true", "pos": "0 0 0"},
        )
        # Outer glow
        ET.SubElement(
            body,
            "geom",
            {
                "name": "ee_marker_glow",
                "type": "sphere",
                "size": "0.008",
                "rgba": "0 1 0.2 0.3",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )
        # Inner core
        ET.SubElement(
            body,
            "geom",
            {
                "name": "ee_marker_geom",
                "type": "sphere",
                "size": "0.005",
                "rgba": "0 1 0.3 1",
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
            },
        )


def _resolved_meshdir(root: ET.Element) -> str:
    compiler = root.find("compiler")
    if compiler is None or not compiler.get("meshdir"):
        return str((_MJCF_PATH.parent / "stl").resolve())
    meshdir_path = Path(compiler.get("meshdir"))
    if not meshdir_path.is_absolute():
        meshdir_path = (_MJCF_PATH.parent / meshdir_path).resolve()
    return str(meshdir_path)


def _load_mj_model() -> "mujoco.MjModel":
    tree = ET.parse(_MJCF_PATH)
    root = tree.getroot()
    _inject_scene(root)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", _resolved_meshdir(root))

    with tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False) as tmp:
        tree.write(tmp, encoding="utf-8")
        tmp_path = Path(tmp.name)
    try:
        return mujoco.MjModel.from_xml_path(str(tmp_path))
    except ValueError:
        model_stl_dir = _REPO_ROOT / "model" / "stl"
        if not model_stl_dir.is_dir():
            raise
        compiler.set("meshdir", str(model_stl_dir))
        with tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False) as tmp2:
            tree.write(tmp2, encoding="utf-8")
            tmp_path_2 = Path(tmp2.name)
        try:
            return mujoco.MjModel.from_xml_path(str(tmp_path_2))
        finally:
            tmp_path_2.unlink(missing_ok=True)
    finally:
        tmp_path.unlink(missing_ok=True)


def _get_joint_qpos_indices(model, joint_names: list[str]) -> list[int]:
    indices = []
    for name in joint_names:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id == -1:
            raise RuntimeError(f"Joint '{name}' not found in MJCF model")
        indices.append(model.jnt_qposadr[jnt_id])
    return indices


def _get_body_pos(model, data, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


def _set_home_pose(
    model,
    data,
    left_qpos: list[int],
    right_qpos: list[int],
    left_jaw_idx: int,
    right_jaw_idx: int,
) -> None:
    for idx, val in zip(left_qpos, _HOME_POSE_RAD):
        data.qpos[idx] = val
    for idx, val in zip(right_qpos, _HOME_POSE_RAD):
        data.qpos[idx] = val
    data.qpos[left_jaw_idx] = _JAW_OPEN
    data.qpos[right_jaw_idx] = _JAW_OPEN
    data.qvel[:] = 0
    if data.act.size:
        data.act[:] = 0
    mujoco.mj_forward(model, data)


def _set_mocap_pos(model, data, body_name: str, world_xyz: np.ndarray):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        return
    mocap_id = model.body_mocapid[body_id]
    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = world_xyz


def _set_cube_pos(model, data, world_xyz: np.ndarray):
    _set_mocap_pos(model, data, "target_cube", world_xyz)


def _update_ee_marker(model, data, ee_body: str):
    _set_mocap_pos(model, data, "ee_marker", _get_body_pos(model, data, ee_body))


# ---------------------------------------------------------------------------
# Animation helpers
# ---------------------------------------------------------------------------


def _interpolate(
    trajectory: list[np.ndarray], min_steps: int = 200
) -> list[np.ndarray]:
    if len(trajectory) >= min_steps or len(trajectory) < 2:
        return trajectory
    orig = np.array(trajectory)
    t_orig = np.linspace(0, 1, len(orig))
    t_new = np.linspace(0, 1, min_steps)
    interp = np.zeros((min_steps, orig.shape[1]))
    for j in range(orig.shape[1]):
        interp[:, j] = np.interp(t_new, t_orig, orig[:, j])
    return [interp[i] for i in range(min_steps)]


def _stabilize(model, data):
    """Zero velocities/accelerations and run forward kinematics for a jitter-free frame."""
    data.qvel[:] = 0
    data.qacc[:] = 0
    if data.act.size:
        data.act[:] = 0
    mujoco.mj_forward(model, data)


def _animate_trajectory(
    viewer,
    model,
    data,
    trajectory,
    qpos_indices,
    speed,
    ik_solver=None,
    arm=None,
    cube_track_body=None,
):
    """Animate joint trajectory. Tracks EE via Pinocchio FK and optionally moves cube with gripper."""
    dt = max(3.0 / len(trajectory), 0.01) / speed
    for q_step in trajectory:
        if not viewer.is_running():
            return
        for idx, q_val in zip(qpos_indices, q_step):
            data.qpos[idx] = q_val
        _stabilize(model, data)
        if ik_solver and arm:
            ee_world = ik_solver.ee_world_pos(q_step, arm=arm)
            _set_mocap_pos(model, data, "ee_marker", ee_world)
        if cube_track_body:
            _set_cube_pos(model, data, _get_body_pos(model, data, cube_track_body))
        viewer.sync()
        time.sleep(dt)


def _animate_jaw(viewer, model, data, jaw_idx, start, end, steps, speed):
    """Smoothly animate a jaw joint from start to end angle."""
    dt = max(1.0 / steps, 0.005) / speed
    for val in np.linspace(start, end, steps):
        if not viewer.is_running():
            return
        data.qpos[jaw_idx] = val
        _stabilize(model, data)
        viewer.sync()
        time.sleep(dt)


def _hold(viewer, model, data, seconds):
    """Hold current pose for a duration."""
    _stabilize(model, data)
    viewer.sync()
    t0 = time.time()
    while viewer.is_running() and time.time() - t0 < seconds:
        viewer.sync()
        time.sleep(0.05)


def _table_z_in_base_frame(ik: IK_SO101) -> float:
    """Z value in the left-arm base frame that places the cube on the table surface."""
    # Table top in world is at _TABLE_Z. Cube center sits at _TABLE_Z + _CUBE_HALF.
    cube_world_z = _TABLE_Z + _CUBE_HALF
    return cube_world_z - ik._base_t[2]


def _sample_reachable_target(
    ik: IK_SO101,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> tuple[np.ndarray, np.ndarray, str, list[np.ndarray]] | None:
    """Sample a random target on the table in world coords, choose the best arm, return (base_target, base2_target, arm, trajectory)."""
    # Sample in world frame so targets span both arms' workspaces.
    # Table center is at _TABLE_POS; arms are at world y ≈ -0.11 (left) and +0.11 (right).
    world_x_bounds = (-0.55, -0.25)  # in front of robot (negative world X)
    world_y_bounds = (-0.20, 0.20)  # spans both arms
    world_z = _TABLE_Z + _CUBE_HALF  # cube on table surface

    for _ in range(max_attempts):
        target_world = np.array(
            [
                rng.uniform(*world_x_bounds),
                rng.uniform(*world_y_bounds),
                world_z,
            ]
        )
        # Convert to both arm base frames
        target_base = ik._base_R.T @ (target_world - ik._base_t)
        target_base2 = ik._base2_R.T @ (target_world - ik._base2_t)

        arm = ik.choose_arm(target_base, target_base2)
        active_target = target_base if arm == "left" else target_base2
        traj = ik.generate_ik_bimanual(
            active_target.tolist(), arm=arm, seed_q_rad=_HOME_POSE_RAD
        )
        if traj:
            return target_base, target_base2, arm, traj
    return None


# ---------------------------------------------------------------------------
# Main visualization loop
# ---------------------------------------------------------------------------


def run_visualization(
    cube_base: list[float] | None = None,
    speed: float = 1.0,
):
    ik = IK_SO101()
    rng = np.random.default_rng()

    print(f"Loading MJCF model from {_MJCF_PATH} ...")
    model = _load_mj_model()
    data = mujoco.MjData(model)

    left_qpos = _get_joint_qpos_indices(model, _LEFT_JOINT_NAMES)
    right_qpos = _get_joint_qpos_indices(model, _RIGHT_JOINT_NAMES)
    left_jaw_idx = _get_joint_qpos_indices(model, [_LEFT_JAW])[0]
    right_jaw_idx = _get_joint_qpos_indices(model, [_RIGHT_JAW])[0]

    _set_home_pose(model, data, left_qpos, right_qpos, left_jaw_idx, right_jaw_idx)

    print("Launching MuJoCo viewer... (close the window to exit)")
    print("Pipeline: place cube -> choose arm -> approach -> reset")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY
        time.sleep(0.5)

        loop_count = 0
        while viewer.is_running():
            loop_count += 1

            # --- Reset both arms to home pose, jaws open ---
            _set_home_pose(
                model, data, left_qpos, right_qpos, left_jaw_idx, right_jaw_idx
            )
            _update_ee_marker(model, data, _LEFT_EE_BODY)
            viewer.sync()

            # ----------------------------------------------------------
            # 1. Place cube
            # ----------------------------------------------------------
            if cube_base is not None:
                target_base = np.asarray(cube_base, dtype=float)
                target_world = ik.base_to_world(target_base)
                target_base2 = ik._base2_R.T @ (target_world - ik._base2_t)
                arm = ik.choose_arm(target_base, target_base2)
                active_target = target_base if arm == "left" else target_base2
                approach_traj = ik.generate_ik_bimanual(
                    active_target.tolist(), arm=arm, seed_q_rad=_HOME_POSE_RAD
                )
                if not approach_traj:
                    print(f"  Loop {loop_count}: IK failed for fixed target, skipping.")
                    time.sleep(1.0)
                    continue
            else:
                result = _sample_reachable_target(ik, rng)
                if result is None:
                    print(
                        f"  Loop {loop_count}: Failed to sample reachable target, retrying."
                    )
                    continue
                target_base, target_base2, arm, approach_traj = result

            # Show cube in viewer
            cube_world = ik.base_to_world(target_base)
            _set_cube_pos(model, data, cube_world)
            _stabilize(model, data)
            viewer.sync()

            if arm == "left":
                arm_qpos = left_qpos
                jaw_idx = left_jaw_idx
                ee_body = _LEFT_EE_BODY
                active_target = target_base
            else:
                arm_qpos = right_qpos
                jaw_idx = right_jaw_idx
                ee_body = _RIGHT_EE_BODY
                active_target = target_base2

            # Show where Pinocchio thinks the EE ends up
            final_ee = ik.ee_world_pos(approach_traj[-1], arm=arm)
            print(
                f"  Loop {loop_count}: cube at [{target_base[0]:.3f}, {target_base[1]:.3f}, {target_base[2]:.3f}] "
                f"-> {arm} arm, {len(approach_traj)} IK steps"
            )
            print(
                f"    EE     (world): [{final_ee[0]:.4f}, {final_ee[1]:.4f}, {final_ee[2]:.4f}]"
            )
            print(
                f"    Cube   (world): [{cube_world[0]:.4f}, {cube_world[1]:.4f}, {cube_world[2]:.4f}]"
                f"  error: {np.linalg.norm(final_ee - cube_world) * 1000:.1f} mm"
            )

            # ----------------------------------------------------------
            # 2. Pause to show the cube before moving
            # ----------------------------------------------------------
            # ----------------------------------------------------------
            # 3. Approach cube
            # ----------------------------------------------------------
            # Prepend current arm pose so interpolation starts from where the arm is
            current_q = np.array([data.qpos[i] for i in arm_qpos])
            approach_traj.insert(0, current_q)
            approach_smooth = _interpolate(approach_traj, min_steps=200)
            _animate_trajectory(
                viewer,
                model,
                data,
                approach_smooth,
                arm_qpos,
                speed,
                ik_solver=ik,
                arm=arm,
            )
            if not viewer.is_running():
                break
            _hold(viewer, model, data, 1.0)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo pick pipeline visualizer",
    )
    parser.add_argument(
        "--cube-x",
        type=float,
        default=None,
        help="Target cube X in Base frame. Omit for random placement.",
    )
    parser.add_argument(
        "--cube-y",
        type=float,
        default=None,
        help="Target cube Y in Base frame (-Y is forward). Omit for random.",
    )
    parser.add_argument(
        "--cube-z",
        type=float,
        default=None,
        help="Target cube Z in Base frame. Omit for random.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier. Default: 1.0",
    )

    args = parser.parse_args()

    if all(v is not None for v in [args.cube_x, args.cube_y, args.cube_z]):
        cube_base = [args.cube_x, args.cube_y, args.cube_z]
    elif any(v is not None for v in [args.cube_x, args.cube_y, args.cube_z]):
        parser.error("Specify all of --cube-x, --cube-y, --cube-z or none for random.")
    else:
        cube_base = None

    run_visualization(cube_base=cube_base, speed=args.speed)


if __name__ == "__main__":
    main()
