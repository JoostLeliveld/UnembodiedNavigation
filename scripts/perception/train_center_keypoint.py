#!/usr/bin/env python3
"""Train the one-keypoint robot-centre observation model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_epochs(run_dir: Path) -> int:
    results = run_dir / "results.csv"
    if not results.is_file():
        return 0
    with results.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _finalize(
    run_dir: Path,
    *,
    data: Path,
    base: Path,
    epochs_requested: int,
    imgsz: int,
    batch: int,
    device: str,
    termination: str,
) -> dict:
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise RuntimeError(f"training did not produce {best}")
    shutil.copy2(best, run_dir / "model.pt")
    manifest = {
        "status": "trained_not_yet_runtime_commissioned",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "task": "pose",
        "observation": "full RGB image",
        "target": "one amodal base_link-xy centre keypoint projected at z=0.35 m",
        "dataset_yaml": str(data),
        "dataset_manifest_sha256": _sha256(data.parent / "dataset_manifest.json"),
        "base_model": str(base),
        "base_model_sha256": _sha256(base),
        "model": str(run_dir / "model.pt"),
        "model_sha256": _sha256(run_dir / "model.pt"),
        "epochs_requested": int(epochs_requested),
        "epochs_completed": _completed_epochs(run_dir),
        "termination": termination,
        "image_size": int(imgsz),
        "batch": int(batch),
        "device": str(device),
        "seed": 0,
        "runtime_warning": "do not deploy until held-out pixel/world bias and covariance are commissioned",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--base-model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--finalize-existing", action="store_true",
        help="finalize the best checkpoint in an existing interrupted run without training",
    )
    parser.add_argument(
        "--resume-existing", action="store_true",
        help="resume an interrupted Ultralytics run from its last checkpoint",
    )
    args = parser.parse_args()

    data = args.data.expanduser().resolve()
    base = args.base_model.expanduser().resolve()
    out = args.out.expanduser().resolve()
    if args.finalize_existing and args.resume_existing:
        raise ValueError("--finalize-existing and --resume-existing are mutually exclusive")
    if out.exists() and not (args.finalize_existing or args.resume_existing):
        raise FileExistsError(f"output already exists: {out}")
    if not data.is_file() or not (data.parent / ".complete").is_file():
        raise FileNotFoundError(f"dataset is absent or incomplete: {data.parent}")
    if not base.is_file():
        raise FileNotFoundError(f"base pose model does not exist: {base}")

    if args.finalize_existing:
        if not out.is_dir():
            raise FileNotFoundError(f"training run does not exist: {out}")
        manifest = _finalize(
            out, data=data, base=base, epochs_requested=args.epochs,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            termination="manually_stopped_after_complete_validation_checkpoint",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    from ultralytics import YOLO

    if args.resume_existing:
        last = out / "weights" / "last.pt"
        if not last.is_file():
            raise FileNotFoundError(f"resume checkpoint does not exist: {last}")
        model = YOLO(str(last))
        result = model.train(resume=True)
        run_dir = Path(getattr(result, "save_dir", out)).resolve()
        manifest = _finalize(
            run_dir, data=data, base=base, epochs_requested=args.epochs,
            imgsz=args.imgsz, batch=args.batch, device=args.device,
            termination="completed_scheduled_training_after_resume",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    model = YOLO(str(base))
    result = model.train(
        task="pose",
        data=str(data),
        epochs=int(args.epochs),
        imgsz=int(args.imgsz),
        batch=int(args.batch),
        device=str(args.device),
        workers=int(args.workers),
        patience=int(args.patience),
        project=str(out.parent),
        name=out.name,
        exist_ok=False,
        pretrained=True,
        deterministic=True,
        seed=0,
        plots=True,
        close_mosaic=min(5, max(int(args.epochs) // 4, 1)),
        cache=False,
        verbose=True,
    )
    run_dir = Path(getattr(result, "save_dir", out)).resolve()
    manifest = _finalize(
        run_dir, data=data, base=base, epochs_requested=args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=args.device,
        termination="completed_scheduled_training",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
