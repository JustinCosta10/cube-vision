"""IK solver for bimanual SO-101 and joint mapping utilities."""

import time

import numpy as np
import pinocchio as pin
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

from cube_vision import MJCF_PATH, RAD2DEG

try:
    from pinocchio.visualize import MeshcatVisualizer
    import meshcat.geometry as g
    import meshcat.transformations as tf
except ModuleNotFoundError:
    MeshcatVisualizer = None

_LEFT_ARM_JOINTS = ["Rotation_L", "Pitch_L", "Elbow_L", "Wrist_Pitch_L", "Wrist_Roll_L"]
_RIGHT_ARM_JOINTS = ["Rotation_R", "Pitch_R", "Elbow_R", "Wrist_Pitch_R", "Wrist_Roll_R"]
_ARM_JOINTS = set(_LEFT_ARM_JOINTS + _RIGHT_ARM_JOINTS)


# ---------------------------------------------------------------------------
# Joint mapping: MJCF <-> motor degrees
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# IK solver
# ---------------------------------------------------------------------------


class IK_SO101:
    _SEED_CONFIGS_DEG = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 90.0, 90.0, 0.0, 0.0],
        [0.0, 135.0, 135.0, 45.0, 0.0],
    ]

    def __init__(self) -> None:
        full_model = pin.buildModelFromMJCF(str(MJCF_PATH))
        q_neutral = pin.neutral(full_model)
        lock_ids = [
            i for i in range(1, full_model.njoints)
            if full_model.names[i] not in _ARM_JOINTS
        ]
        self.model = pin.buildReducedModel(full_model, lock_ids, q_neutral)
        self.data = self.model.createData()

        self._joint_q_idx = {}
        for i in range(1, self.model.njoints):
            self._joint_q_idx[self.model.names[i]] = self.model.joints[i].idx_q

        self._left_q_indices = np.array([self._joint_q_idx[j] for j in _LEFT_ARM_JOINTS])
        self._right_q_indices = np.array([self._joint_q_idx[j] for j in _RIGHT_ARM_JOINTS])
        self.EE_LEFT = "Gripper_Tip"
        self.EE_RIGHT = "Gripper_Tip_2"

        q = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        base_oMf = self.data.oMf[self.model.getFrameId("Base")]
        self._base_R = base_oMf.rotation.copy()
        self._base_t = base_oMf.translation.copy()
        base2_oMf = self.data.oMf[self.model.getFrameId("Base_2")]
        self._base2_R = base2_oMf.rotation.copy()
        self._base2_t = base2_oMf.translation.copy()

        self.dt = 0.01
        self.q = pin.neutral(self.model)
        self.configuration = pink.Configuration(self.model, self.data, self.q)

        for wrist_roll_joint in ["Wrist_Roll_L", "Wrist_Roll_R"]:
            idx = self._joint_q_idx[wrist_roll_joint]
            self.model.lowerPositionLimit[idx] = 0.0
            self.model.upperPositionLimit[idx] = 0.0

        self.ee_left_task = FrameTask(self.EE_LEFT, position_cost=10.0, orientation_cost=0.0)
        self.ee_right_task = FrameTask(self.EE_RIGHT, position_cost=10.0, orientation_cost=0.0)
        self.posture_task = PostureTask(cost=1e-4)
        self.tasks = [self.ee_left_task, self.ee_right_task, self.posture_task]
        self._elbow_up_q_5 = np.deg2rad([0.0, 90.0, 90.0, 0.0, 0.0])

    def base_to_world(self, p_base: np.ndarray) -> np.ndarray:
        return self._base_R @ np.asarray(p_base) + self._base_t

    def base2_to_world(self, p_base2: np.ndarray) -> np.ndarray:
        return self._base2_R @ np.asarray(p_base2) + self._base2_t

    def choose_arm(self, target_base_xyz: np.ndarray, target_base2_xyz: np.ndarray) -> str:
        dist_left = np.linalg.norm(target_base_xyz)
        dist_right = np.linalg.norm(target_base2_xyz)
        chosen = "left" if dist_left <= dist_right else "right"
        print(f"Arm selection: left dist={dist_left:.4f}, right dist={dist_right:.4f} -> {chosen}")
        return chosen

    def ee_world_pos(self, q_arm: np.ndarray, arm: str = "left") -> np.ndarray:
        """Run FK on 5 arm joint angles (rad) and return the EE position in world frame."""
        q = pin.neutral(self.model)
        indices = self._left_q_indices if arm == "left" else self._right_q_indices
        q[indices] = np.asarray(q_arm, dtype=float)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        ee_name = self.EE_LEFT if arm == "left" else self.EE_RIGHT
        return self.data.oMf[self.model.getFrameId(ee_name)].translation.copy()

    def _get_current_ee_world_pos(self, ee_frame_name: str) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, self.configuration.q)
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.model.getFrameId(ee_frame_name)].translation.copy()

    def _build_seed_q(self, seed_deg_5: list[float], arm: str) -> np.ndarray:
        q_seed = pin.neutral(self.model)
        seed_rad = np.deg2rad(seed_deg_5)
        if arm == "left":
            q_seed[self._left_q_indices] = seed_rad
        else:
            q_seed[self._right_q_indices] = seed_rad
        return np.clip(q_seed, self.model.lowerPositionLimit, self.model.upperPositionLimit)

    def _run_ik_from_seed(
        self,
        q_seed: np.ndarray,
        active_ee_frame: str,
        target_transform: "pin.SE3",
        idle_target: "pin.SE3",
        position_tolerance: float,
        max_timesteps: int,
    ) -> tuple[list[np.ndarray], float]:
        self.configuration = pink.Configuration(self.model, self.data, q_seed.copy())
        active_task = self.ee_left_task if active_ee_frame == self.EE_LEFT else self.ee_right_task
        idle_task = self.ee_right_task if active_ee_frame == self.EE_LEFT else self.ee_left_task
        active_task.set_target(target_transform)
        idle_task.set_target(idle_target)
        active_task.position_cost = 10.0
        idle_task.position_cost = 100.0

        posture_q = pin.neutral(self.model)
        posture_q[self._left_q_indices] = self._elbow_up_q_5
        posture_q[self._right_q_indices] = self._elbow_up_q_5
        self.posture_task.set_target(posture_q)

        trajectory: list[np.ndarray] = []
        ee_frame_id = self.model.getFrameId(active_ee_frame)
        for step in range(max_timesteps):
            pin.forwardKinematics(self.model, self.data, self.configuration.q)
            pin.updateFramePlacements(self.model, self.data)
            pos_error = target_transform.translation - self.data.oMf[ee_frame_id].translation
            if np.linalg.norm(pos_error) < position_tolerance:
                break
            try:
                dq = solve_ik(self.configuration, self.tasks, self.dt, solver="quadprog")
            except Exception as exc:
                print(f"IK Solver Failed at Step{step}. Error: {exc}")
                break
            self.configuration.integrate_inplace(dq, self.dt)
            trajectory.append(self.configuration.q.copy())

        pin.forwardKinematics(self.model, self.data, self.configuration.q)
        pin.updateFramePlacements(self.model, self.data)
        final_error = np.linalg.norm(
            target_transform.translation - self.data.oMf[ee_frame_id].translation
        )
        return trajectory, final_error

    def generate_ik_bimanual(
        self,
        target_xyz: list[float],
        arm: str = "left",
        gripper_offset_xyz: list[float] | None = None,
        position_tolerance: float = 1e-3,
        max_timesteps: int = 1000,
        seed_q_rad: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        offset = np.asarray(gripper_offset_xyz) if gripper_offset_xyz else np.zeros(3)
        base_xyz = np.asarray(target_xyz) + offset

        if arm == "left":
            world_xyz = self.base_to_world(base_xyz)
            active_ee = self.EE_LEFT
            idle_ee = self.EE_RIGHT
            active_indices = self._left_q_indices
        else:
            world_xyz = self.base2_to_world(base_xyz)
            active_ee = self.EE_RIGHT
            idle_ee = self.EE_LEFT
            active_indices = self._right_q_indices

        target_transform = pin.SE3(np.eye(3), world_xyz)
        idle_world_pos = self._get_current_ee_world_pos(idle_ee)
        idle_target = pin.SE3(np.eye(3), idle_world_pos)

        best_traj: list[np.ndarray] = []
        best_error = float("inf")
        seeds = []
        if seed_q_rad is not None:
            q_custom = self.configuration.q.copy()
            q_custom[active_indices] = np.asarray(seed_q_rad, dtype=float)
            seeds.append(np.clip(q_custom, self.model.lowerPositionLimit, self.model.upperPositionLimit))
        for seed_deg in self._SEED_CONFIGS_DEG:
            seeds.append(self._build_seed_q(seed_deg, arm))

        for q_seed in seeds:
            traj, error = self._run_ik_from_seed(
                q_seed,
                active_ee,
                target_transform,
                idle_target,
                position_tolerance,
                max_timesteps,
            )
            if error < best_error:
                best_error = error
                best_traj = traj
            if best_error < position_tolerance:
                break

        if len(best_traj) >= max_timesteps:
            print(f"IK did not converge: error={best_error*1000:.1f}mm, steps={len(best_traj)}/{max_timesteps}")
            return []
        if best_error >= position_tolerance:
            print(f"IK failed to reach tolerance: error={best_error*1000:.1f}mm, tol={position_tolerance*1000:.1f}mm")
            return []

        if best_traj:
            self.q = best_traj[-1].copy()
            self.configuration = pink.Configuration(self.model, self.data, self.q)
        return [q[active_indices] for q in best_traj]

    def generate_ik(
        self,
        target_xyz: list[float],
        gripper_offset_xyz: list[float],
        position_tolerance: float = 1e-3,
        max_timesteps: int = 1000,
        seed_q_rad: np.ndarray | None = None,
    ):
        return self.generate_ik_bimanual(
            target_xyz,
            arm="left",
            gripper_offset_xyz=gripper_offset_xyz,
            position_tolerance=position_tolerance,
            max_timesteps=max_timesteps,
            seed_q_rad=seed_q_rad,
        )

    def visualize_ik(self, trajectory: list, object_xyz):
        if MeshcatVisualizer is None:
            print("Meshcat failed to import.")
            return
        collision_model = pin.GeometryModel()
        visual_model = pin.GeometryModel()
        viz = MeshcatVisualizer(self.model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
        viz.display(self.q)

        viewer = viz.viewer
        viewer["target_cube"].set_object(g.Box([0.017, 0.017, 0.017]), g.MeshLambertMaterial(color=0x00FFFF, opacity=0.8))
        viewer["target_cube"].set_transform(tf.translation_matrix(np.array(object_xyz)))
        ee_frame_id = self.model.getFrameId(self.EE_LEFT)
        viz.viewer["ee_point"].set_object(g.Sphere(0.005), g.MeshLambertMaterial(color=0xFF0000))
        viz.viewer["ee_point"].set_transform(tf.translation_matrix(self.data.oMf[ee_frame_id].translation))

        for q_step in trajectory:
            viz.display(q_step)
            pin.forwardKinematics(self.model, self.data, q_step)
            pin.updateFramePlacements(self.model, self.data)
            time.sleep(self.dt)
            viz.viewer["ee_point"].set_transform(
                tf.translation_matrix(self.data.oMf[ee_frame_id].translation)
            )
