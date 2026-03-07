# cube-vision

Refactored into a package-first layout:

- `cube_vision`: reusable library code
- `scripts`: runnable entrypoints and manual checks
- `tests`: automated unit tests only
- `model`, `calibration`: robot assets and calibration data

Common entrypoints:

- `python control.py`
- `python calibrate.py --bus all`
- `python -m pytest`
