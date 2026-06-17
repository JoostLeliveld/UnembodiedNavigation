#!/usr/bin/env python3
# [DEPRECATED_LEGACY_CLEANUP] Legacy/exploratory/diagnostic script or module. Distracting from paper-facing F85-F88 runtime.
"""Entry point: YOLO training pipeline.

Steps (run each script directly):
  1. scripts/perception/capture_external_camera_images.py
  2. scripts/perception/make_redmask_pseudolabels.py
  3. scripts/perception/train_yolo_seg.py
  4. scripts/perception/test_yolo_out_of_box.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'shared'))

if __name__ == '__main__':
    print("Run YOLO training pipeline steps individually — see docstring for order.")
