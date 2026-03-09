"""Cube Vision — robot vision and manipulation for SO-101."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

DEG2RAD = 3.141592653589793 / 180.0
RAD2DEG = 180.0 / 3.141592653589793

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_CONFIG_PATH = REPO_ROOT / "config.yaml"
_config_cache: dict | None = None


def load_config(path: Path | None = None) -> dict:
    """Load and cache config.yaml. Returns the parsed dict."""
    global _config_cache
    if _config_cache is not None and path is None:
        return _config_cache
    p = path or _CONFIG_PATH
    with open(p) as f:
        cfg = yaml.safe_load(f)
    if path is None:
        _config_cache = cfg
    return cfg


# ---------------------------------------------------------------------------
# Derived paths
# ---------------------------------------------------------------------------

CALIBRATION_DIR = REPO_ROOT / "calibration"
OUTPUTS_DIR = REPO_ROOT / "outputs"
REALSENSE_OUTPUT_DIR = OUTPUTS_DIR / "realsense_capture"

_cfg = load_config()
MJCF_PATH = REPO_ROOT / _cfg["model"]["mjcf"]
