from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / 'scripts' / 'perception'
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_half_open_bbox_dataset import build_dataset, half_open_box, yolo_detection_row


def test_half_open_box_uses_max_plus_one() -> None:
    mask = np.zeros((10, 12), dtype=np.uint8)
    mask[2:5, 3:8] = 255

    assert half_open_box(mask) == (3.0, 2.0, 8.0, 5.0)
    assert yolo_detection_row(half_open_box(mask), width=12, height=10) == (
        '0 0.45833333 0.35000000 0.41666667 0.30000000\n'
    )


def test_build_dataset_preserves_split_and_writes_detection_labels(tmp_path: Path) -> None:
    source = tmp_path / 'source'
    masks = tmp_path / 'masks'
    for split in ('train', 'val'):
        (source / 'images' / split).mkdir(parents=True)
        (masks / split).mkdir(parents=True)
        image = np.zeros((10, 12, 3), dtype=np.uint8)
        mask = np.zeros((10, 12), dtype=np.uint8)
        if split == 'train':
            mask[2:5, 3:8] = 255
        assert cv2.imwrite(str(source / 'images' / split / f'{split}.png'), image)
        assert cv2.imwrite(str(masks / split / f'{split}.png'), mask)
    (source / 'data.yaml').write_text(
        yaml.safe_dump({
            'path': str(source),
            'train': 'images/train',
            'val': 'images/val',
            'names': {0: 'robot'},
            'task': 'segment',
        }, sort_keys=False),
        encoding='utf-8',
    )

    output = tmp_path / 'derived'
    manifest = build_dataset(source / 'data.yaml', masks, output)

    assert manifest['splits']['train'] == {'images': 1, 'positives': 1, 'negatives': 0}
    assert manifest['splits']['val'] == {'images': 1, 'positives': 0, 'negatives': 1}
    assert (output / 'labels/train/train.txt').read_text(encoding='utf-8').startswith('0 ')
    assert (output / 'labels/val/val.txt').read_text(encoding='utf-8') == ''
    assert yaml.safe_load((output / 'data.yaml').read_text(encoding='utf-8'))['task'] == 'detect'
    assert (output / '.complete').is_file()

    with pytest.raises(FileExistsError):
        build_dataset(source / 'data.yaml', masks, output)
