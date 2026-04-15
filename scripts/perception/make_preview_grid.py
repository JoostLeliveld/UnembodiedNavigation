#!/usr/bin/env python3
"""Create a single preview grid image from a folder of preview images."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')


def main() -> int:
    parser = argparse.ArgumentParser(description='Create a simple preview grid from a folder of images.')
    parser.add_argument('--input', required=True, help='Folder containing preview images')
    parser.add_argument('--out', required=True, help='Output preview grid PNG path')
    parser.add_argument('--cols', type=int, default=3)
    parser.add_argument('--limit', type=int, default=12)
    args = parser.parse_args()

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.is_dir():
        raise RuntimeError(f'Input folder does not exist: {input_dir}')
    out_path = Path(args.out).expanduser().resolve()
    images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in sorted(input_dir.rglob('*')) if path.suffix.lower() in IMAGE_EXTS]
    images = [img for img in images if img is not None][: max(int(args.limit), 1)]
    if not images:
        raise RuntimeError(f'No preview images found in {input_dir}')

    h = max(img.shape[0] for img in images)
    w = max(img.shape[1] for img in images)
    canvases = []
    for img in images:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        canvases.append(canvas)

    rows = []
    n_cols = max(int(args.cols), 1)
    for start in range(0, len(canvases), n_cols):
        chunk = canvases[start:start + n_cols]
        while len(chunk) < n_cols:
            chunk.append(np.zeros((h, w, 3), dtype=np.uint8))
        rows.append(np.concatenate(chunk, axis=1))
    grid = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    print(f'Wrote preview grid to {out_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
