from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cube_vision.config.paths import OUTPUTS_DIR


def save_ik_plot(
    base_pos: np.ndarray,
    ik_target_base: np.ndarray,
    camera_centroid_cam: np.ndarray | None = None,
    camera_pos_world: np.ndarray | None = None,
    ee_pos_base: np.ndarray | None = None,
    filename: str = "ik_target.png",
):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=-0.02, right=0.82, bottom=0.16, top=0.96)
    t = np.asarray(ik_target_base)

    ax.scatter(0, 0, 0, c="black", s=100, marker="s", label="Base origin")
    axis_len = 0.10
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="red", arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="green", arrow_length_ratio=0.15, linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="blue", arrow_length_ratio=0.15, linewidth=2)
    ax.text(axis_len * 1.1, 0, 0, "X", color="red", fontsize=10)
    ax.text(0, axis_len * 1.1, 0, "Y", color="green", fontsize=10)
    ax.text(0, 0, axis_len * 1.1, "Z", color="blue", fontsize=10)

    ax.scatter(t[0], t[1], t[2], c="magenta", s=150, marker="*", label="IK target")
    ax.plot([0, t[0]], [0, t[1]], [0, t[2]], "m--", alpha=0.4)

    if ee_pos_base is not None:
        ee = np.asarray(ee_pos_base)
        ax.scatter(ee[0], ee[1], ee[2], c="cyan", s=100, marker="^", label="EE position")

    info_lines = [
        f"IK target (Base): [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]",
        f"Distance from base: {np.linalg.norm(t):.4f} m",
        f"Base world pos: [{base_pos[0]:.4f}, {base_pos[1]:.4f}, {base_pos[2]:.4f}]",
    ]
    if camera_centroid_cam is not None:
        c = np.asarray(camera_centroid_cam)
        info_lines.append(f"Cam centroid (optical): [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
    if ee_pos_base is not None:
        ee = np.asarray(ee_pos_base)
        info_lines.append(f"EE (Base): [{ee[0]:.4f}, {ee[1]:.4f}, {ee[2]:.4f}]")
    fig.text(
        0.50,
        0.02,
        "\n".join(info_lines),
        fontsize=8,
        family="monospace",
        verticalalignment="bottom",
        horizontalalignment="center",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("IK Target in Base Frame")
    ax.legend(loc="upper right")

    all_pts = [np.zeros(3), t]
    if ee_pos_base is not None:
        all_pts.append(np.asarray(ee_pos_base))
    all_pts = np.array(all_pts)
    mid = (all_pts.max(axis=0) + all_pts.min(axis=0)) / 2
    span = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max(), 0.15) / 2 * 1.2
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    out_path = OUTPUTS_DIR / filename
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"Saved IK visualization to {out_path}")
    return str(out_path)
