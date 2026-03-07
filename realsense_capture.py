#!/usr/bin/env python3
"""Compatibility wrapper for RealSense capture."""

from cube_vision.hardware.realsense import capture

__all__ = ["capture"]

if __name__ == "__main__":
    capture()
