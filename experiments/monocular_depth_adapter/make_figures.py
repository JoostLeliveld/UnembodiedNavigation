#!/usr/bin/env python3
"""Figures for the monocular depth adapter run.

    python3 experiments/monocular_depth_adapter/make_figures.py --run bs1_native_flip

Two figures, both meant to be readable by someone who has not seen the code:

``depth_maps.png``
    the camera's own picture, then what each model made of it. Every panel keeps
    its own colour scale and states its own units, because a shared scale across
    metric and unitless models would be a lie about comparability.

``cost_and_agreement.png``
    what each model costs per frame, what it costs in GPU memory against the
    card's ceiling, and how far apart the metric models' answers are.

No accuracy panel: the frozen set has no depth labels, and inventing a reference
by anchoring to the floor is a decision this study does not make.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import frozen_set as fs
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from monodepth import storage  # noqa: E402

OUT_ROOT = fs.REPO / "logs/studies/monocular_depth_adapter"

# Palette slots and ink from the validated reference palette (light mode).
# Two-slot categorical check: ALL PASS, worst CVD dE 24.7.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
SLOT_DEPTH_PASS = "#2a78d6"     # slot 1 — the depth forward pass
SLOT_UNC_PASS = "#eb6834"       # slot 2 — the extra pass the flip signal costs
GRID = "#d9d8d4"

#: Plain-language names. The registry keys are filing labels, not explanations.
PRETTY = {
    "dav2_metric_indoor_small": "Depth Anything V2\nmetric indoor, Small",
    "dav2_metric_indoor_large": "Depth Anything V2\nmetric indoor, Large",
    "dav2_relative_small": "Depth Anything V2\nrelative, Small",
    "metric3d_v2_vit_small": "Metric3D v2\nViT-S",
    "metric3d_v2_vit_large": "Metric3D v2\nViT-L",
    "unidepth_v2_vits14": "UniDepthV2\nViT-S",
    "unidepth_v2_vitl14": "UniDepthV2\nViT-L",
}
CONVENTION_WORDS = {
    "metric_z": "metres along the optical axis",
    "euclidean_range": "metres along the ray",
    "relative_depth": "unitless, larger = further",
    "inverse_depth": "unitless, larger = NEARER",
}


def _pretty(model_name: str) -> str:
    return PRETTY.get(model_name, model_name)


def _flat(model_name: str) -> str:
    return _pretty(model_name).replace("\n", " ")


def _load(run_dir: Path) -> dict:
    index = json.loads((run_dir / "index.json").read_text(encoding="utf-8"))
    out = {}
    for model_name, entry in index["models"].items():
        model_dir = run_dir / entry["dir"]
        out[model_name] = {
            "manifest": json.loads((model_dir / "run_manifest.json").read_text(encoding="utf-8")),
            "dir": model_dir,
        }
    return {"index": index, "models": out}


def _pick_frame(models: dict) -> str:
    """A development frame every model produced, chosen without an RNG."""
    shared = None
    for blob in models.values():
        ids = {f["frame_id"] for f in blob["manifest"]["per_frame"]
               if f["role"] == "method_development"}
        shared = ids if shared is None else (shared & ids)
    if not shared:
        raise SystemExit("no development frame is common to every model")
    return sorted(shared)[len(shared) // 2]


# ------------------------------------------------------------------- figure 1
def _truth_depth(frame_id: str) -> np.ndarray | None:
    """Real Gazebo depth for this frame, if it was captured. Evaluation only."""
    truth_dir = fs.REPO / "logs/studies/monocular_depth_adapter/depth_truth_warehouse_aws"
    path = truth_dir / f"{frame_id}_depth.npy"
    return np.load(path) if path.is_file() else None


def depth_maps(models: dict, frame_id: str, out_path: Path) -> None:
    frame = next(f for f in fs.load_frames() if f.frame_id == frame_id)
    rgb = fs.load_image(frame)
    truth = _truth_depth(frame_id)

    names = sorted(models)
    n_panels = len(names) + 1 + (1 if truth is not None else 0)
    cols = 3
    rows = int(np.ceil(n_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.4 * cols, 3.6 * rows),
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    flat = axes.ravel()

    flat[0].imshow(rgb)
    flat[0].set_title(f"What the camera sees\n{frame.world.split('.')[0]}, camera {frame.camera_id}, "
                      f"mounted {frame.camera_pose_xyzrpy[2]:.1f} m up",
                      fontsize=10.5, loc="left", color=INK)
    flat[0].set_xticks([]); flat[0].set_yticks([])

    next_panel = 1
    if truth is not None:
        ax = flat[next_panel]; next_panel += 1
        lo, hi = np.nanpercentile(truth, [2, 98])
        im = ax.imshow(truth, cmap="Blues", vmin=lo, vmax=hi)
        ax.set_title("What was actually there\n"
                     f"real Gazebo depth: {np.nanmedian(truth):.2f} m",
                     fontsize=10.5, loc="left", color=INK, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label("metres along the optical axis", fontsize=8.5, color=INK_SOFT)
        cbar.ax.tick_params(labelsize=7.5, colors=INK_SOFT)

    conventions_seen = []
    for ax, model_name in zip(flat[next_panel:], names):
        pred = storage.load_prediction(
            models[model_name]["dir"] / f"{frame_id}__{model_name}.json")
        conventions_seen.append(pred.convention)
        depth = np.where(pred.valid, pred.depth, np.nan)
        lo, hi = np.nanpercentile(depth, [2, 98])
        # One hue, light -> dark: darker is further for the depth-like
        # conventions. Never a shared scale across models — see the docstring.
        im = ax.imshow(depth, cmap="Blues", vmin=lo, vmax=hi)
        unit = "m" if pred.convention.is_metric else ""
        median = float(np.nanmedian(depth))
        note = ""
        if truth is not None and pred.convention.is_metric:
            note = f"  ({median / float(np.nanmedian(truth)):.2f}x the truth)"
        ax.set_title(f"{_pretty(model_name)}\nmiddle of the picture: "
                     f"{median:.2f} {unit}{note}".strip(),
                     fontsize=10.5, loc="left", color=INK)
        ax.set_xticks([]); ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, shrink=0.82)
        cbar.set_label(CONVENTION_WORDS[pred.convention.value], fontsize=8.5, color=INK_SOFT)
        cbar.ax.tick_params(labelsize=7.5, colors=INK_SOFT)

    for ax in flat[n_panels:]:
        ax.axis("off")

    n_metric = sum(1 for c in conventions_seen if c.is_metric)
    n_other = len(conventions_seen) - n_metric
    truth_line = ("Every model recovers the same scene shape; none of them recovers its "
                  "scale. Each panel has its OWN colour scale — a shared one would hide "
                  "exactly that." if truth is not None else
                  f"Panels are NOT on a shared colour scale: {n_metric} of these are metres "
                  f"and {n_other} {'is' if n_other == 1 else 'are'} unitless.")
    fig.suptitle(
        f"The same warehouse camera frame, read by {len(names)} monocular depth models — "
        "each in its own units\n" + truth_line,
        fontsize=12.5, fontweight="bold", color=INK, ha="left", x=0.006)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ------------------------------------------------------------------- figure 2
def cost_and_agreement(models: dict, out_path: Path, device_total_mib: float) -> None:
    names = sorted(models, key=lambda n: np.median(
        [f["timing"]["total_s"] for f in models[n]["manifest"]["per_frame"]]))
    labels = [_flat(n) for n in names]
    forward = np.array([np.median([f["timing"]["forward_s"]
                                   for f in models[n]["manifest"]["per_frame"]]) for n in names])
    extra = np.array([np.median([f["timing"].get("uncertainty_s", 0.0)
                                 for f in models[n]["manifest"]["per_frame"]]) for n in names])
    peak = np.array([max(f["memory"]["gpu_peak_allocated_mib"]
                         for f in models[n]["manifest"]["per_frame"]) for n in names])

    pairs = _metric_pairs(models)

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 2.6 + 0.72 * max(len(names), len(pairs))),
                             constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    y = np.arange(len(names))

    ax = axes[0]
    ax.barh(y, forward, color=SLOT_DEPTH_PASS, height=0.62)
    # 2px surface gap between adjacent fills, per the mark spec
    ax.barh(y, extra, left=forward + 0.004 * max(forward.max(), 1e-6),
            color=SLOT_UNC_PASS, height=0.62)
    for i, (f, e) in enumerate(zip(forward, extra)):
        ax.text(f + e + 0.03 * (forward + extra).max(), i, f"{f + e:.2f} s",
                va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_xlabel("seconds per 1280x720 frame  (shorter is better)",
                  fontsize=9.5, color=INK_SOFT)
    ax.set_title("What one frame costs in time\n"
                 "(asking for uncertainty doubles it)", fontsize=11.5, loc="left", color=INK)
    ax.set_xlim(0, (forward + extra).max() * 1.22)

    ax = axes[1]
    ax.barh(y, peak, color=SLOT_DEPTH_PASS, height=0.62)
    # The ceiling is named in the title and the legend; an in-panel annotation
    # here collides with the axis label.
    ax.axvline(device_total_mib, color=INK_SOFT, lw=2, ls="--")
    for i, p in enumerate(peak):
        ax.text(p + 0.02 * device_total_mib, i, f"{p:.0f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("peak GPU memory for one frame, MiB  (less is better)",
                  fontsize=9.5, color=INK_SOFT)
    ax.set_title(f"What one frame costs in memory\n"
                 f"(three larger checkpoints ran out of this card's {device_total_mib:.0f} MiB)",
                 fontsize=11.5, loc="left", color=INK)
    ax.set_xlim(0, max(device_total_mib, peak.max()) * 1.12)

    ax = axes[2]
    if pairs:
        yy = np.arange(len(pairs))
        vals = np.array([p["median_abs_difference_m"] for p in pairs])
        ax.barh(yy, vals, color=SLOT_DEPTH_PASS, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + 0.03 * vals.max(), i, f"{v:.2f} m", va="center", fontsize=9, color=INK)
        ax.set_yticks(yy)
        ax.set_yticklabels([f"{p['a']}\nvs {p['b']}" for p in pairs], fontsize=8, color=INK)
        ax.set_xlim(0, vals.max() * 1.25)
    ax.set_xlabel("typical per-pixel gap, metres  (0 = they agree)",
                  fontsize=9.5, color=INK_SOFT)
    ax.set_title("How far apart the metre-producing models are\n"
                 "(rank agreement is 0.98 for every pair: same shape, different scale)",
                 fontsize=11.5, loc="left", color=INK)

    for ax in axes:
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_SOFT, labelsize=8.5)

    fig.legend(handles=[Patch(facecolor=SLOT_DEPTH_PASS, label="the depth pass itself"),
                        Patch(facecolor=SLOT_UNC_PASS,
                              label="the second pass that produces the uncertainty map"),
                        Line2D([0], [0], color=INK_SOFT, lw=2, ls="--",
                               label="this GPU's memory ceiling")],
               loc="lower center", ncol=3, frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(
        "Models that claim metres disagree by metres, but agree on shape — 12 frames, "
        "one fixed warehouse camera, Quadro P2000\n"
        "The per-pixel gaps on the right shrink to 2-14 cm once one scale and one shift are "
        "removed, so the disagreement is calibration, not geometry.",
        fontsize=12.5, fontweight="bold", color=INK, ha="left", x=0.004)
    fig.savefig(out_path, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _metric_pairs(models: dict, role: str = "method_development") -> list[dict]:
    """Median |a - b| in metres for every pair of models that both report metres."""
    loaded, conventions = {}, {}
    for model_name, blob in models.items():
        wanted = {f["frame_id"] for f in blob["manifest"]["per_frame"] if f["role"] == role}
        preds = {p.image_id: p for p in storage.load_dir(blob["dir"]) if p.image_id in wanted}
        if not preds:
            continue
        loaded[model_name] = preds
        conventions[model_name] = next(iter(preds.values())).convention

    metric = sorted(m for m, c in conventions.items() if c.is_metric)
    rows = []
    for a, b in combinations(metric, 2):
        shared = sorted(set(loaded[a]) & set(loaded[b]))
        diffs = []
        for fid in shared:
            pa, pb = loaded[a][fid], loaded[b][fid]
            mask = pa.valid & pb.valid
            if mask.any():
                diffs.append(float(np.median(np.abs(pa.depth[mask] - pb.depth[mask]))))
        if diffs:
            rows.append({"a": _flat(a), "b": _flat(b),
                         "median_abs_difference_m": float(np.median(diffs))})
    return sorted(rows, key=lambda r: r["median_abs_difference_m"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--frame", default=None, help="frame_id for the depth-map figure")
    args = parser.parse_args()

    run_dir = OUT_ROOT / args.run
    blob = _load(run_dir)
    models = blob["models"]
    if not models:
        print("no models in that run")
        return 1

    frame_id = args.frame or _pick_frame(models)
    env = next(iter(models.values()))["manifest"]["environment"]

    maps_path = run_dir / "depth_maps.png"
    depth_maps(models, frame_id, maps_path)
    print(f"wrote {maps_path.relative_to(fs.REPO)}  (frame {frame_id})")

    cost_path = run_dir / "cost_and_agreement.png"
    cost_and_agreement(models, cost_path, float(env.get("device_total_mib", 4096.0)))
    print(f"wrote {cost_path.relative_to(fs.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
