#!/usr/bin/env python3
"""Entry point: full observability GP capture pipeline.

Steps (run each script directly):
  1. scripts/visibility_comparison/capture_visibility_samples.py
  2. scripts/visibility_comparison/extract_perception_targets.py
  3. scripts/visibility_comparison/build_gp_targets.py
  4. scripts/visibility_comparison/fit_visibility_gps.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'shared'))

if __name__ == '__main__':
    print("Run observability pipeline steps individually — see docstring for order.")
