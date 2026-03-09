# Cube Vision

Vision-guided pick pipeline for bimanual SO-101 arms.

## Setup

```bash
git clone <repo-url>
cd cube-vision
pip install -e .
```

## Configuration

Edit `config.yaml` to match your hardware — bus ports, motor IDs, camera settings, PID tuning.

## Scripts

**Run the pick demo** (requires robot hardware + RealSense):
```bash
python scripts/run_pick_demo.py
```

**Visualize the pipeline in MuJoCo** (no hardware needed):
```bash
python scripts/visualize_mujoco.py               # random cube positions
python scripts/visualize_mujoco.py --cube-x 0.0 --cube-y -0.25 --cube-z 0.02
python scripts/visualize_mujoco.py --speed 2.0    # faster playback
```

**Calibrate motors** (requires hardware):
```bash
python -m cube_vision.hardware
```

## Tests

```bash
pytest
```

## Project Structure

```
config.yaml              # hardware config — edit this for your setup
cube_vision/
    __init__.py           # paths, constants, config loader
    hardware.py           # motor definitions, calibration, RealSense capture
    vision.py             # color detection, point cloud
    transforms.py         # camera-to-base frame transforms (Pinocchio FK)
    ik.py                 # inverse kinematics solver (bimanual SO-101)
    visualize.py          # debug visualizations (color detection overlay, IK plot)
    pick.py               # pick-by-color workflow
scripts/
    run_pick_demo.py      # main entry point for the pick pipeline
    visualize_mujoco.py   # MuJoCo simulation of the full pipeline
    manual/               # diagnostic scripts (require human + hardware)
```
