#!/usr/bin/env python3
"""MuJoCo visualizer for the xlerobot IK pipeline.

Simulates the full control pipeline (frame_transform + IK solver) on an
imaginary cube on a table, without requiring hardware.

Usage:
    python visualize_mujoco.py                              # default cube 25cm from arm on table
    python visualize_mujoco.py --cube-x 0.05 --cube-y -0.25 --cube-z 0.02
    python visualize_mujoco.py --use-transform              # run frame_transform pipeline
    python visualize_mujoco.py --speed 0.5                  # slower playback
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
# On macOS, MuJoCo's launch_passive requires the script to run under mjpython.
# Auto-relaunch if we detect we're on macOS and not already under mjpython.
# ---------------------------------------------------------------------------
if platform.system() == "Darwin" and "MJPYTHON" not in os.environ:
    mjpython = shutil.which("mjpython")
    if mjpython:
        env = os.environ.copy()
        env["MJPYTHON"] = "1"  # prevent infinite re-launch
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

_MJCF_PATH = _REPO_ROOT / "model" / "xlerobot.xml"

# Offset converts IK world positions (URDF) to MJCF world positions.
# URDF Base X=-0.135, MJCF Base X=-0.09; URDF Base Y=-0.088, MJCF Base Y=-0.11
_MJCF_OFFSET = np.array([0.045, -0.022, 0.015])

# The 5 IK joints in the order returned by generate_ik()
_IK_JOINT_NAMES = [
    "Rotation_L",
    "Pitch_L",
    "Elbow_L",
    "Wrist_Pitch_L",
    "Wrist_Roll_L",
]

def _ensure_target_cube_body(root: ET.Element) -> None:
    """Ensure the MJCF contains a visible mocap cube body named target_cube."""
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    for body in worldbody.findall("body"):
        if body.get("name") == "target_cube":
            return

    body = ET.SubElement(
        worldbody,
        "body",
        {"name": "target_cube", "mocap": "true", "pos": "0 0 0.15"},
    )
    ET.SubElement(
        body,
        "geom",
        {
            "name": "target_cube_geom",
            "type": "box",
            "size": "0.017 0.017 0.017",
            "rgba": "1 0.1 0.1 1",
            "contype": "0",
            "conaffinity": "0",
            "group": "1",
        },
    )


def _resolved_meshdir(root: ET.Element) -> str:
    """Return an absolute meshdir for robust temporary-XML loading."""
    compiler = root.find("compiler")
    if compiler is None:
        return str((_MJCF_PATH.parent / "stl").resolve())

    meshdir = compiler.get("meshdir")
    if not meshdir:
        return str((_MJCF_PATH.parent / "stl").resolve())

    meshdir_path = Path(meshdir)
    if not meshdir_path.is_absolute():
        meshdir_path = (_MJCF_PATH.parent / meshdir_path).resolve()
    return str(meshdir_path)


def _load_mj_model_with_mesh_fallback() -> "mujoco.MjModel":
    """Load MJCF, falling back to repo-local model/stl if meshdir is broken."""
    tree = ET.parse(_MJCF_PATH)
    root = tree.getroot()
    _ensure_target_cube_body(root)
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    compiler.set("meshdir", _resolved_meshdir(root))

    with tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False) as tmp:
        tree.write(tmp, encoding="utf-8")
        tmp_xml_path = Path(tmp.name)

    try:
        return mujoco.MjModel.from_xml_path(str(tmp_xml_path))
    except ValueError as exc:
        model_stl_dir = _REPO_ROOT / "model" / "stl"
        if not model_stl_dir.is_dir():
            raise
        print(f"MJCF load failed: {exc}")
        print(f"Retrying with meshdir={model_stl_dir}")
        compiler.set("meshdir", str(model_stl_dir))
        with tempfile.NamedTemporaryFile("wb", suffix=".xml", delete=False) as tmp2:
            tree.write(tmp2, encoding="utf-8")
            tmp_xml_path_2 = Path(tmp2.name)
        try:
            return mujoco.MjModel.from_xml_path(str(tmp_xml_path_2))
        finally:
            tmp_xml_path_2.unlink(missing_ok=True)
    finally:
        tmp_xml_path.unlink(missing_ok=True)


def _resolve_joint_qpos_indices(model: "mujoco.MjModel") -> list[int]:
    """Map each IK joint name to its qpos index in the MuJoCo model."""
    indices = []
    for name in _IK_JOINT_NAMES:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jnt_id == -1:
            raise RuntimeError(f"Joint '{name}' not found in MJCF model")
        indices.append(model.jnt_qposadr[jnt_id])
    return indices


def _update_target_cube(model, data, world_xyz: np.ndarray):
    """Move the mocap target cube to the given world-frame position."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cube")
    if body_id == -1:
        return
    # mocap bodies have their own index
    mocap_id = model.body_mocapid[body_id]
    if mocap_id >= 0:
        data.mocap_pos[mocap_id] = world_xyz


def _interpolate_trajectory(trajectory: list[np.ndarray], min_anim_steps: int = 200) -> list[np.ndarray]:
    """Upsample short IK trajectories so motion is visibly smooth."""
    if len(trajectory) >= min_anim_steps or len(trajectory) < 2:
        return trajectory
    orig = np.array(trajectory)
    t_orig = np.linspace(0, 1, len(orig))
    t_new = np.linspace(0, 1, min_anim_steps)
    interp = np.zeros((min_anim_steps, orig.shape[1]))
    for j in range(orig.shape[1]):
        interp[:, j] = np.interp(t_new, t_orig, orig[:, j])
    return [interp[i] for i in range(min_anim_steps)]


def _sample_random_reachable_target(
    ik: IK_SO101,
    gripper_offset: list[float],
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> tuple[np.ndarray, list[np.ndarray]] | None:
    """Sample a random base-frame cube location and keep only IK-reachable targets."""
    # Conservative reachable workspace in base frame (meters).
    x_bounds = (-0.12, 0.12)
    y_bounds = (-0.34, -0.14)  # forward is negative Y in this project
    z_bounds = (-0.02, 0.16)

    for _ in range(max_attempts):
        target = np.array(
            [
                rng.uniform(*x_bounds),
                rng.uniform(*y_bounds),
                rng.uniform(*z_bounds),
            ],
            dtype=float,
        )
        trajectory = ik.generate_ik(
            target_xyz=target.tolist(),
            gripper_offset_xyz=gripper_offset,
        )
        if trajectory:
            return target, trajectory
    return None


def run_visualization(
    cube_base: list[float],
    gripper_offset: list[float],
    use_transform: bool = False,
    speed: float = 1.0,
):
    """Run the full IK pipeline and animate in MuJoCo viewer."""

    # ------------------------------------------------------------------
    # Optionally run the frame_transform pipeline with a synthetic point
    # ------------------------------------------------------------------
    if use_transform:
        from cube_vision.transforms import camera_xyz_to_base_xyz

        # Synthesize a camera-frame point that roughly maps to our target.
        # Use head at neutral (pan=0, tilt=0 in motor convention = pan≈1°, tilt≈14° motor).
        # We just pass 0,0 in radians as a simple demo.
        joint_values = {"head_pan_joint": 0.0, "head_tilt_joint": 0.0}

        # A point 30cm forward, slightly down in camera optical frame
        cam_x, cam_y, cam_z = 0.0, 0.05, 0.30
        bx, by, bz = camera_xyz_to_base_xyz(cam_x, cam_y, cam_z, joint_values)
        cube_base = [bx, by, bz]
        print(f"[frame_transform] Camera ({cam_x}, {cam_y}, {cam_z}) "
              f"-> Base ({bx:.4f}, {by:.4f}, {bz:.4f})")

    # ------------------------------------------------------------------
    # Set up IK solver
    # ------------------------------------------------------------------
    ik = IK_SO101()
    rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Load MuJoCo model
    # ------------------------------------------------------------------
    print(f"Loading MJCF model from {_MJCF_PATH} ...")
    model = _load_mj_model_with_mesh_fallback()
    data = mujoco.MjData(model)

    qpos_indices = _resolve_joint_qpos_indices(model)

    # Set initial pose and forward
    mujoco.mj_forward(model, data)

    # ------------------------------------------------------------------
    # Animate in passive viewer
    # ------------------------------------------------------------------
    print("Launching MuJoCo viewer... (close the window to exit)")
    print("  Label toggle keys in viewer:  i = body/link  j = joint  u = off")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # -- Enable labels & frames for joints and links -----------------
        # mjtLabel: 0=none, 1=body, 2=joint, 3=geom, 4=site …
        # mjtFrame: 0=none, 1=body, 2=geom, 3=site, 4=world, 6=joint
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_BODY   # show body/link names
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_BODY    # show body coord frames

        # Give the viewer a moment to initialize
        time.sleep(0.5)

        # Loop: animate trajectory, hold 8s, reset, repeat
        loop_count = 0
        while viewer.is_running():
            loop_count += 1
            print(f"Animation loop {loop_count}...")

            if use_transform:
                loop_cube_base = np.asarray(cube_base, dtype=float)
                trajectory = ik.generate_ik(
                    target_xyz=loop_cube_base.tolist(),
                    gripper_offset_xyz=gripper_offset,
                )
                if not trajectory:
                    print("WARNING: Fixed target became unreachable; skipping this loop.")
                    continue
            else:
                sampled = _sample_random_reachable_target(ik, gripper_offset, rng=rng)
                if sampled is None:
                    print("WARNING: Failed to sample reachable random target; skipping this loop.")
                    continue
                loop_cube_base, trajectory = sampled

            trajectory = _interpolate_trajectory(trajectory, min_anim_steps=200)
            final_q = trajectory[-1]
            print(
                f"  Target Base: [{loop_cube_base[0]:.3f}, {loop_cube_base[1]:.3f}, {loop_cube_base[2]:.3f}] "
                f"| IK steps: {len(trajectory)} "
                f"| Final joints deg: {np.rad2deg(final_q).round(1).tolist()}"
            )

            target_world_ik = ik.base_to_world(np.asarray(loop_cube_base) + np.asarray(gripper_offset))
            target_world = target_world_ik + _MJCF_OFFSET
            _update_target_cube(model, data, target_world)

            # Target ~3 seconds for full animation at speed=1
            dt_display = max(3.0 / len(trajectory), 0.01) / speed

            # Animate through trajectory
            for step_i, q_step in enumerate(trajectory):
                if not viewer.is_running():
                    break
                for idx, q_val in zip(qpos_indices, q_step):
                    data.qpos[idx] = q_val
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(dt_display)

            # Hold final pose for 8 seconds
            hold_start = time.time()
            while viewer.is_running() and time.time() - hold_start < 8.0:
                _update_target_cube(model, data, target_world)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.05)

            # Reset to neutral before next loop
            if viewer.is_running():
                for idx in qpos_indices:
                    data.qpos[idx] = 0.0
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.5)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo visualizer for xlerobot IK pipeline",
    )
    parser.add_argument(
        "--cube-x", type=float, default=0.0,
        help="Target cube X in Base frame (left/right). Default: 0.0",
    )
    parser.add_argument(
        "--cube-y", type=float, default=-0.25,
        help="Target cube Y in Base frame (-Y is forward). Default: -0.20",
    )
    parser.add_argument(
        "--cube-z", type=float, default=0.0,
        help="Target cube Z in Base frame (up/down). Default: 0.0",
    )
    parser.add_argument(
        "--gripper-offset-x", type=float, default=0.0,
        help="Gripper offset X in Base frame. Default: 0.0",
    )
    parser.add_argument(
        "--gripper-offset-y", type=float, default=0.0,
        help="Gripper offset Y in Base frame. Default: 0.0",
    )
    parser.add_argument(
        "--gripper-offset-z", type=float, default=0.0,
        help="Gripper offset Z in Base frame. Default: 0.0",
    )
    parser.add_argument(
        "--use-transform", action="store_true",
        help="Run frame_transform pipeline with synthetic camera point",
    )
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier. Default: 1.0",
    )

    args = parser.parse_args()

    cube_base = [args.cube_x, args.cube_y, args.cube_z]
    gripper_offset = [args.gripper_offset_x, args.gripper_offset_y, args.gripper_offset_z]

    run_visualization(
        cube_base=cube_base,
        gripper_offset=gripper_offset,
        use_transform=args.use_transform,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
