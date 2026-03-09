"""Visualization helpers for color detection and IK targets."""

import json
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cube_vision import DEG2RAD, OUTPUTS_DIR, REALSENSE_OUTPUT_DIR
from cube_vision.vision import COLOR_RANGES, detect_color, detection_to_xyz
from cube_vision.transforms import camera_xyz_to_base2_xyz, camera_xyz_to_base_xyz

# ---------------------------------------------------------------------------
# Color detection visualization
# ---------------------------------------------------------------------------


def visualize_color_detection(
    color: str = "red",
    head_pan_deg: float = 0.0,
    head_tilt_deg: float = 0.0,
    captures_dir: Path = REALSENSE_OUTPUT_DIR,
    out_name: str = "color_detect_vis.png",
    show_window: bool = False,
    window_ms: int = 1200,
    exclude_bottom_fraction: float = 0.0,
):
    bgr = cv2.imread(str(captures_dir / "color.png"))
    if bgr is None:
        raise FileNotFoundError(f"color.png not found in {captures_dir}")
    depth_m = np.load(captures_dir / "depth_meters.npy")
    with open(captures_dir / "intrinsic_data.json") as f:
        intrinsics = json.load(f)

    joint_values = {
        "head_pan_joint": head_pan_deg * DEG2RAD,
        "head_tilt_joint": head_tilt_deg * DEG2RAD,
    }

    detections = detect_color(bgr, color, exclude_bottom_fraction=exclude_bottom_fraction)
    vis = bgr.copy()
    if exclude_bottom_fraction > 0.0:
        h = vis.shape[0]
        cutoff = max(0, min(h, int(h * (1.0 - exclude_bottom_fraction))))
        cv2.line(vis, (0, cutoff), (vis.shape[1] - 1, cutoff), (0, 0, 255), 2)
        cv2.putText(vis, f"Bottom {exclude_bottom_fraction*100:.0f}% ignored", (10, max(20, cutoff - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    if not detections:
        cv2.putText(vis, f"No {color} objects found", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        for i, det in enumerate(detections):
            x, y, w, h = det.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.drawContours(vis, [det.contour], -1, (255, 255, 0), 1)
            cx, cy = det.centroid_px
            cv2.circle(vis, (cx, cy), 6, (0, 0, 255), -1)
            cv2.circle(vis, (cx, cy), 8, (255, 255, 255), 2)

            try:
                xyz = detection_to_xyz(det, depth_m, intrinsics)
                cam_label = f"cam:({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f})"
                bx, by, bz = camera_xyz_to_base_xyz(xyz[0], xyz[1], xyz[2], joint_values)
                b2x, b2y, b2z = camera_xyz_to_base2_xyz(xyz[0], xyz[1], xyz[2], joint_values)
                arm_label = f"base:({bx:.3f},{by:.3f},{bz:.3f})"
                arm2_label = f"base2:({b2x:.3f},{b2y:.3f},{b2z:.3f})"
            except RuntimeError:
                cam_label = "cam: no depth"
                arm_label = ""
                arm2_label = ""

            cv2.putText(vis, f"#{i+1} {cam_label}", (x, y - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
            if arm_label:
                cv2.putText(vis, arm_label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 2)
            if arm2_label:
                cv2.putText(vis, arm2_label, (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 80), 2)

        cv2.putText(vis, f"Detected {len(detections)} {color} object(s)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    out_path = captures_dir / out_name
    cv2.imwrite(str(out_path), vis)
    print(f"Saved visualization to {out_path}")
    if not show_window:
        return
    try:
        if window_ms > 0:
            cv2.imshow("Color Detection", vis)
            cv2.waitKey(window_ms)
            cv2.destroyWindow("Color Detection")
        else:
            raise cv2.error("highgui", "imshow", "Persistent mode requested", -1)
    except cv2.error as exc:
        print(f"OpenCV popup unavailable, trying eog fallback: {exc}")
        eog = shutil.which("eog")
        if eog is None:
            print("eog not found on PATH, skipping popup display.")
            return
        proc = subprocess.Popen([eog, str(out_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if window_ms > 0:
            time.sleep(window_ms / 1000.0)
            if proc.poll() is None:
                proc.terminate()
        else:
            print("Opened visualization in eog (left running).")


# ---------------------------------------------------------------------------
# IK target plot
# ---------------------------------------------------------------------------


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
