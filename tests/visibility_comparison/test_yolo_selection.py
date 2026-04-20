from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
for rel in ('src/perception',):
    path = str((REPO_ROOT / rel).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)

from perception.core.yolo_selection import select_best_detection


class _Tensor:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self._values, dtype=float)


class _Boxes:
    def __init__(self, cls: list[int], conf: list[float], xyxy: list[list[float]]):
        self.cls = _Tensor(cls)
        self.conf = _Tensor(conf)
        self.xyxy = _Tensor(xyxy)

    def __len__(self):
        return int(self.conf.numpy().shape[0])


class _Result:
    def __init__(self):
        self.boxes = _Boxes(
            cls=[0],
            conf=[0.18],
            xyxy=[[10.0, 20.0, 30.0, 60.0]],
        )
        self.masks = None


def test_select_best_detection_preserves_subthreshold_raw_score() -> None:
    selected = select_best_detection(
        _Result(),
        confidence_threshold=0.25,
        target_ids={0},
        use_masks=True,
        mask_min_area=0.0,
        mask_bottom_band_px=3,
    )

    assert selected['detected'] is False
    assert selected['detected_after_threshold'] is False
    assert selected['raw_best_score'] == 0.18
    assert selected['selected_score'] == 0.18
    assert selected['best_class_id'] == 0
    assert selected['selected_pixel_source'] == 'bbox_bottom'
