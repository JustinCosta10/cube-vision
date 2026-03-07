from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CALIBRATION_DIR = REPO_ROOT / "calibration"
OUTPUTS_DIR = REPO_ROOT / "outputs"
REALSENSE_OUTPUT_DIR = OUTPUTS_DIR / "realsense_capture"
MODEL_DIR = REPO_ROOT / "model"
MJCF_PATH = MODEL_DIR / "xlerobot.xml"
