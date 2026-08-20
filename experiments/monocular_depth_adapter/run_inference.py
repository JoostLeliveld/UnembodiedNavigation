#!/usr/bin/env python3
"""Run one or more depth models over the frozen image set and save the results.

    # everything, default settings
    python3 experiments/monocular_depth_adapter/run_inference.py

    # one model, four-camera batch, flip-consistency uncertainty
    python3 experiments/monocular_depth_adapter/run_inference.py \
        --models unidepth_v2_vits14 --batch-size 4 --uncertainty native+flip

Predictions land in ``logs/studies/monocular_depth_adapter/<run>/<model>/`` as
npz + json pairs, with a ``run_manifest.json`` per model recording the exact
configuration, per-frame runtime and memory, and any batch that had to be split
because the GPU ran out.

Nothing here compares against ground truth. It cannot: the adapter has no path
to it, and the frozen set carries no depth labels.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import sys
import time
from pathlib import Path

# Set before torch initialises CUDA. On a 4 GB card the largest checkpoints sit
# right at the ceiling, and the default allocator's fixed segments fragment
# enough across a multi-model run to turn a model that fits into one that does
# not. Recorded in the manifest so a memory figure can always be traced to the
# allocator it was measured under.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import frozen_set as fs
import numpy as np

from monodepth import MonocularDepthAdapter, available_models, storage, uncertainty as unc

OUT_ROOT = fs.REPO / "logs/studies/monocular_depth_adapter"


def _gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def _free_gpu() -> None:
    """Hand the card back between models so the next one starts from a clean slate."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _environment() -> dict:
    import torch

    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "device_name": _gpu_name(),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["device_total_mib"] = props.total_memory / 2 ** 20
        info["device_capability"] = f"{props.major}.{props.minor}"
    return info


def run_model(model_name: str, frames, out_dir: Path, *, batch_size: int,
              uncertainty_mode: str, warmup: int, device: str,
              add_temporal: bool) -> dict:
    """Run one model over the frames and write everything it produced."""
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = fs.to_requests(frames)

    started = time.time()
    adapter = MonocularDepthAdapter(model_name, device=device, batch_size=batch_size,
                                    uncertainty=uncertainty_mode)
    load_t0 = time.perf_counter()
    adapter.load()
    load_s = time.perf_counter() - load_t0

    # Warm-up passes are excluded from the recorded timings: the first call pays
    # for cuDNN autotuning and lazy kernel loading, which is a one-off cost and
    # would otherwise dominate a 12-frame average.
    for _ in range(warmup):
        adapter.predict(requests[:min(batch_size, len(requests))])

    predictions = adapter.predict(requests)
    info = adapter.info

    per_frame = []
    for frame, pred in zip(frames, predictions):
        storage.save_prediction(pred, out_dir)
        finite = pred.depth[pred.valid]
        per_frame.append({
            "frame_id": pred.image_id,
            "camera_id": frame.camera_id,
            "world": frame.world,
            "role": frame.role,
            "convention": pred.convention.value,
            "shape": list(pred.depth.shape),
            "valid_fraction": pred.valid_fraction,
            "depth_min": float(finite.min()) if finite.size else float("nan"),
            "depth_median": float(np.median(finite)) if finite.size else float("nan"),
            "depth_max": float(finite.max()) if finite.size else float("nan"),
            "timing": pred.timing.as_dict(),
            "memory": pred.memory.as_dict(),
            "uncertainty_kind": pred.uncertainty_kind,
            "uncertainty": unc.summarize(pred.uncertainty),
            "extras": dict(pred.extras),
        })

    temporal: dict = {"computed": False}
    if add_temporal:
        temporal = _temporal_block(frames, predictions, out_dir)

    adapter.unload()

    manifest = {
        "model": info.as_dict(),
        "frozen_set": {
            "name": fs.DEFAULT_SET,
            "n_frames": len(frames),
            "frame_ids": [f.frame_id for f in frames],
        },
        "config": {
            "batch_size": batch_size,
            "uncertainty_mode": uncertainty_mode,
            "warmup_batches": warmup,
            "device": device,
            "seed": adapter.seed,
        },
        "determinism": adapter.determinism_config,
        "environment": _environment(),
        "load_seconds": load_s,
        "wall_seconds": time.time() - started,
        "oom_events": adapter.oom_events,
        "per_frame": per_frame,
        "temporal_disagreement": temporal,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _temporal_block(frames, predictions, out_dir: Path) -> dict:
    """Per-camera spread across frames — only where the camera really is fixed."""
    by_camera: dict[tuple, list] = {}
    for frame, pred in zip(frames, predictions):
        by_camera.setdefault((frame.world, frame.camera_id), []).append(pred)

    result: dict = {"computed": True, "cameras": {}}
    for (world, camera_id), preds in sorted(by_camera.items()):
        if len(preds) < 2:
            continue
        spread, detail = unc.temporal_disagreement(preds)
        key = f"{world.split('.')[0]}/{camera_id}"
        np.savez_compressed(out_dir / f"temporal_spread__{key.replace('/', '_')}.npz",
                            spread=spread)
        result["cameras"][key] = detail
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=available_models(),
                        help=f"default: all of {available_models()}")
    parser.add_argument("--frozen-set", default=fs.DEFAULT_SET)
    parser.add_argument("--role", default=None,
                        choices=["method_development", "batch_plumbing_only"],
                        help="restrict to one role of frame")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--uncertainty", default="native+flip")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-name", default=None,
                        help="output subdirectory; default is derived from the settings")
    parser.add_argument("--no-temporal", action="store_true",
                        help="skip the per-camera temporal spread")
    parser.add_argument("--skip-failing", action="store_true",
                        help="record and continue when a model cannot run")
    args = parser.parse_args()

    problems = fs.verify(args.frozen_set)
    if problems:
        print(f"frozen set {args.frozen_set} failed verification:")
        for p in problems:
            print("  -", p)
        return 1

    frames = fs.load_frames(args.frozen_set, role=args.role)
    run_name = args.run_name or f"bs{args.batch_size}_{args.uncertainty.replace('+', '_')}"
    run_dir = OUT_ROOT / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"frozen set {args.frozen_set}: {len(frames)} frames -> {run_dir.relative_to(fs.REPO)}")

    # Merge rather than overwrite: the biggest checkpoints only fit when they get
    # a process to themselves, so a complete run is often several invocations.
    index_path = run_dir / "index.json"
    index = {"run_name": run_name, "frozen_set": args.frozen_set, "role_filter": args.role,
             "models": {}, "failures": {}}
    if index_path.is_file():
        previous = json.loads(index_path.read_text(encoding="utf-8"))
        index["models"].update(previous.get("models", {}))
        index["failures"].update(previous.get("failures", {}))

    for model_name in args.models:
        print(f"\n=== {model_name} ===", flush=True)
        index["failures"].pop(model_name, None)   # this attempt supersedes the last
        _free_gpu()
        try:
            manifest = run_model(
                model_name, frames, run_dir / model_name,
                batch_size=args.batch_size, uncertainty_mode=args.uncertainty,
                warmup=args.warmup, device=args.device,
                add_temporal=not args.no_temporal,
            )
        except Exception as exc:  # noqa: BLE001 - a model failing is a result, not a crash
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            index["failures"][model_name] = f"{type(exc).__name__}: {exc}"
            if not args.skip_failing:
                raise
            continue

        forwards = [f["timing"]["forward_s"] for f in manifest["per_frame"]]
        peaks = [f["memory"]["gpu_peak_allocated_mib"] for f in manifest["per_frame"]]
        print(f"  convention   {manifest['model']['convention']}")
        print(f"  forward      {np.median(forwards):.3f} s/frame (median, batch {args.batch_size})")
        print(f"  gpu peak     {max(peaks):.0f} MiB   weights {manifest['per_frame'][0]['memory']['weights_mib']:.0f} MiB")
        print(f"  valid pixels {np.mean([f['valid_fraction'] for f in manifest['per_frame']]):.4f} mean")
        if manifest["oom_events"]:
            print(f"  OOM fallbacks: {manifest['oom_events']}")
        index["models"][model_name] = {
            "dir": model_name,
            "convention": manifest["model"]["convention"],
            "median_forward_s": float(np.median(forwards)),
            "max_gpu_peak_mib": float(max(peaks)),
            "oom_events": len(manifest["oom_events"]),
        }

    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nwrote {index_path.relative_to(fs.REPO)} "
          f"({len(index['models'])} model(s) recorded, {len(index['failures'])} failed)")
    return 1 if index["failures"] else 0


if __name__ == "__main__":
    sys.exit(main())
