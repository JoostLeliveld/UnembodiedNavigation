#!/usr/bin/env python3
"""Export a reference-position dataset as a YOLO detector dataset.

Roles: D_mu -> train, D_dev -> val. D_R and final_audit never enter. Each unique image
(image_sha1) is exported once, labelled by the frozen label contract of
`label_dataset_protocol.json` via `export_detector_dataset.classify`. Ambiguous views are
recorded in the ledger and excluded, exactly as in the stage-04 export. A negative frame
that is byte-identical across roles is exported to train only. The dataset is
selected with THESIS_REFERENCE_DATASET.

    THESIS_REFERENCE_DATASET=v7_balanced python3 \
      experiments/thesis_pipeline_lock/export_reference_detector_dataset.py --out DIR
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from experiments.thesis_pipeline_lock.export_detector_dataset import classify  # noqa: E402
from experiments.warehouse_v2_sketches.reference_dataset import (  # noqa: E402
    LOCK_PATH, NAME, image_path, load_rows,
)

SPLITS = {"D_mu": "train", "D_dev": "val"}
PROTOCOL = REPO / "experiments/thesis_pipeline_lock/label_dataset_protocol.json"
WIDTH, HEIGHT = 1280, 720


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(out)
    contract = json.loads(PROTOCOL.read_text(encoding="utf-8"))[
        "detector_label_contract"]["positive_requires_all"]

    rows = [r for r in load_rows() if r["stratum"] in SPLITS]
    if any(r["stratum"] in ("D_R", "final_audit") for r in rows):
        raise RuntimeError("D_R or final_audit entered the detector export")
    for split in SPLITS.values():
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)

    seen: dict[str, str] = {}
    ledger, counts = [], collections.Counter()
    for r in rows:
        label, reasons, box, _ = classify(r, WIDTH, HEIGHT, contract)
        split = SPLITS[r["stratum"]]
        key = r["image_sha1"]
        exported = ""
        if label in ("positive", "negative") and key not in seen:
            seen[key] = split
            stem = f"{key}"
            target = out / "images" / split / f"{stem}.png"
            os.link(image_path(r), target)
            lab = out / "labels" / split / f"{stem}.txt"
            if box is None:
                lab.write_text("", encoding="ascii")
            else:
                x0, y0, x1, y1 = box
                w, h = x1 - x0 + 1, y1 - y0 + 1
                lab.write_text(
                    f"0 {(x0 + x1 + 1) / 2 / WIDTH:.9f} {(y0 + y1 + 1) / 2 / HEIGHT:.9f} "
                    f"{w / WIDTH:.9f} {h / HEIGHT:.9f}\n", encoding="ascii")
            counts[(split, r["camera_id"], label)] += 1
            exported = str(target.relative_to(out))
        elif label in ("positive", "negative") and seen[key] != split:
            # An empty background frame is byte-identical wherever the robot is out of
            # view, so the same negative can occur in both roles. It is kept in train
            # only. A positive can never repeat across roles, since roles are disjoint
            # sets of physical positions.
            if label == "positive":
                raise RuntimeError(f"positive image {key} appears in both train and val")
            reasons = reasons + ["duplicate_of_train_negative"]
        ledger.append({
            "capture_source": r["capture_source"], "plan_pose_index": r["plan_pose_index"],
            "position_key": r["position_key"], "stratum": r["stratum"],
            "camera_id": r["camera_id"], "image_sha1": key, "label_class": label,
            "reasons": ";".join(reasons), "exported_image": exported,
        })

    ledger_path = out / "label_ledger.csv"
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n  0: warehouse_amr_blue\n",
        encoding="utf-8")
    manifest = {
        "schema": "thesis_reference_detector_dataset.v1",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "reference_dataset": NAME,
        "campaign_lock": str(LOCK_PATH.relative_to(REPO)),
        "campaign_lock_sha256": sha256(LOCK_PATH),
        "label_protocol_sha256": sha256(PROTOCOL),
        "export_script_sha256": sha256(Path(__file__).resolve()),
        "splits": SPLITS,
        "final_audit_accessed": False,
        "unique_images_exported": len(seen),
        "counts": [{"split": s, "camera": c, "class": k, "images": n}
                   for (s, c, k), n in sorted(counts.items())],
        "label_ledger_sha256": sha256(ledger_path),
        "dataset_yaml_sha256": sha256(yaml_path),
    }
    (out / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                               encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "counts"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
