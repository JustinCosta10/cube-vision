#!/usr/bin/env python3
"""Compatibility wrapper for color detection helpers."""

import argparse

from cube_vision.perception.color import (
    COLOR_RANGES,
    Detection,
    detect_color,
    detect_object,
    detection_to_xyz,
)

__all__ = ["COLOR_RANGES", "Detection", "detect_color", "detect_object", "detection_to_xyz"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Color-based object detection")
    parser.add_argument("--color", default="red", choices=list(COLOR_RANGES.keys()))
    args = parser.parse_args()
    detect_object(args.color)
