"""Deck-ready showcase of warehouse_v2, its five cameras, and the warehouse AMR.

The five views are deliberately representative clear frames, not a synchronized
multi-camera instant.  That keeps every panel useful as a visual introduction to
the building and makes the robot legible in all five viewpoints.
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import style as D  # noqa: E402


DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"
OUT = D.REPO / "logs/studies/deck_figures/warehouse"

# Fixed, clear commissioning frames chosen for warehouse context, robot
# legibility, and distinct parts of the floor.  Pose differs between panels.
PICKS = {
    "A": "images/train/sample_001043.png",
    "B": "images/train/sample_001150.png",
    "C": "images/train/sample_001555.png",
    "D": "images/train/sample_001626.png",
    "E": "images/val/sample_000948.png",
}

ROBOT_HERO = ("D", "images/val/sample_001790.png")


def diagnostic(camera: str, image: str) -> dict[str, str]:
    path = DATA / f"camera_{camera}" / "label_diagnostics.csv"
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["image"] == image:
                return row
    raise RuntimeError(f"No diagnostics row for camera {camera}: {image}")


def source(camera: str, image: str) -> Path:
    return DATA / f"camera_{camera}" / image


def bbox(row: dict[str, str]) -> tuple[float, float, float, float]:
    x0, y0 = float(row["mask_bbox_x0"]), float(row["mask_bbox_y0"])
    x1, y1 = float(row["mask_bbox_x1"]), float(row["mask_bbox_y1"])
    return x0, y0, x1, y1


def camera_panel(ax, camera: str, image: str) -> None:
    frame = Image.open(source(camera, image)).convert("RGB")
    row = diagnostic(camera, image)
    x0, y0, x1, y1 = bbox(row)
    ax.imshow(frame)
    ax.add_patch(Rectangle((x0 - 8, y0 - 8), x1 - x0 + 16, y1 - y0 + 16,
                           fill=False, edgecolor="white", lw=2.2, zorder=5))
    ax.add_patch(Rectangle((x0 - 8, y0 - 8), x1 - x0 + 16, y1 - y0 + 16,
                           fill=False, edgecolor=D.CAM_COLOUR[camera], lw=1.1,
                           ls=(0, (3, 2)), zorder=6))
    ax.text(0.025, 0.945, f"CAMERA {camera}", transform=ax.transAxes,
            ha="left", va="top", color="white", fontsize=12.5,
            fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=D.CAM_COLOUR[camera],
                      edgecolor="white", linewidth=1.0, alpha=0.96))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(D.CAM_COLOUR[camera])
        spine.set_linewidth(2.2)


def hero_crop(ax) -> None:
    camera, image = ROBOT_HERO
    frame = Image.open(source(camera, image)).convert("RGB")
    row = diagnostic(camera, image)
    x0, y0, x1, y1 = bbox(row)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    crop_w, crop_h = 410.0, 230.0
    left = max(0.0, min(frame.width - crop_w, cx - crop_w / 2))
    top = max(0.0, min(frame.height - crop_h, cy - crop_h / 2))
    crop = frame.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
    ax.imshow(crop)
    ax.add_patch(Rectangle((x0 - left - 8, y0 - top - 8),
                           x1 - x0 + 16, y1 - y0 + 16,
                           fill=False, edgecolor="white", lw=2.4, zorder=5))
    ax.text(0.025, 0.945, "THE NEW AMR", transform=ax.transAxes,
            ha="left", va="top", color="white", fontsize=12.5,
            fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=D.ROBOT,
                      edgecolor="white", linewidth=1.0, alpha=0.96))
    ax.text(0.025, 0.055, "0.80 × 0.55 m  ·  low 0.35 m deck",
            transform=ax.transAxes, ha="left", va="bottom", color="white",
            fontsize=11.3, fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#111111",
                      edgecolor="none", alpha=0.78))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(D.ROBOT)
        spine.set_linewidth(2.2)


def make_five_camera_showcase() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), constrained_layout=False)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.845, bottom=0.095,
                        wspace=0.035, hspace=0.085)
    for ax, (camera, image) in zip(axes.flat[:5], PICKS.items()):
        camera_panel(ax, camera, image)
    hero_crop(axes.flat[5])

    fig.text(0.025, 0.955, "THE NEW WAREHOUSE", fontsize=12.5,
             fontweight="bold", color=D.ROBOT, ha="left", va="top")
    fig.text(0.025, 0.915, "One floor. Five camera viewpoints.", fontsize=25,
             fontweight="bold", color=D.INK, ha="left", va="top")
    fig.text(0.025, 0.865,
             "Warehouse v2 pairs dense storage, working aisles and block stacks with a floor-scale AMR.",
             fontsize=13.2, color=D.INK2, ha="left", va="top")
    fig.text(0.025, 0.035,
             "Representative clear views from the frozen commissioning capture · the robot pose differs between panels",
             fontsize=10.8, color=D.MUTED, ha="left", va="bottom")
    fig.savefig(OUT / "01_five_camera_showcase.png", dpi=180)
    plt.close(fig)


def make_robot_detail() -> None:
    camera, image = ROBOT_HERO
    frame = Image.open(source(camera, image)).convert("RGB")
    row = diagnostic(camera, image)
    x0, y0, x1, y1 = bbox(row)

    fig = plt.figure(figsize=(16.0, 9.0))
    gs = fig.add_gridspec(1, 2, left=0.03, right=0.975, top=0.81, bottom=0.105,
                          width_ratios=(1.65, 1.0), wspace=0.055)
    ax_full = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])

    ax_full.imshow(frame)
    ax_full.add_patch(Rectangle((x0 - 9, y0 - 9), x1 - x0 + 18, y1 - y0 + 18,
                                fill=False, edgecolor="white", lw=3.0, zorder=5))
    ax_full.text(0.025, 0.96, "CAMERA D · FULL VIEW", transform=ax_full.transAxes,
                 ha="left", va="top", color="white", fontsize=12.5,
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.3",
                 facecolor=D.CAM_COLOUR["D"], edgecolor="white", alpha=0.96))

    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    crop_w, crop_h = 350.0, 330.0
    left = max(0.0, min(frame.width - crop_w, cx - crop_w / 2))
    top = max(0.0, min(frame.height - crop_h, cy - crop_h / 2))
    zoom = frame.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
    ax_zoom.imshow(zoom)
    ax_zoom.add_patch(Rectangle((x0 - left - 7, y0 - top - 7),
                                x1 - x0 + 14, y1 - y0 + 14,
                                fill=False, edgecolor="white", lw=2.6))
    ax_zoom.text(0.04, 0.955, "WAREHOUSE AMR", transform=ax_zoom.transAxes,
                 ha="left", va="top", color="white", fontsize=13,
                 fontweight="bold", bbox=dict(boxstyle="round,pad=0.3",
                 facecolor=D.ROBOT, edgecolor="white", alpha=0.96))
    ax_zoom.text(0.04, 0.075,
                 "0.80 × 0.55 m footprint\n0.35 m deck height\nfront sensor bar · rear cabinet",
                 transform=ax_zoom.transAxes, ha="left", va="bottom",
                 color="white", fontsize=13, fontweight="bold", linespacing=1.45,
                 bbox=dict(boxstyle="round,pad=0.55", facecolor="#111111",
                           edgecolor="none", alpha=0.78))

    for ax in (ax_full, ax_zoom):
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#d5d4cf")
            spine.set_linewidth(1.5)

    fig.text(0.03, 0.955, "THE NEW ROBOT", fontsize=12.5, fontweight="bold",
             color=D.ROBOT, ha="left", va="top")
    fig.text(0.03, 0.91, "Built to read as a warehouse AMR from overhead",
             fontsize=25, fontweight="bold", color=D.INK, ha="left", va="top")
    fig.text(0.03, 0.855,
             "The low deck fits the working aisles; the dark front slots and raised rear cabinet preserve heading cues.",
             fontsize=13.2, color=D.INK2, ha="left", va="top")
    fig.text(0.03, 0.035, "Representative Camera D frame from warehouse_v2",
             fontsize=10.8, color=D.MUTED, ha="left", va="bottom")
    fig.savefig(OUT / "02_new_robot_detail.png", dpi=180)
    plt.close(fig)


def copy_camera_frames() -> None:
    raw = OUT / "camera_views"
    raw.mkdir(parents=True, exist_ok=True)
    for camera, image in PICKS.items():
        shutil.copy2(source(camera, image), raw / f"camera_{camera}.png")


def write_source_note() -> None:
    lines = [
        "# Warehouse showcase sources",
        "",
        "The five panels are representative clear commissioning frames, not a synchronized instant.",
        "The robot pose therefore differs between camera views.",
        "",
        "Dataset: `warehouse_v2_yolo_shared_20260822`",
        "",
    ]
    for camera, image in PICKS.items():
        row = diagnostic(camera, image)
        lines.append(
            f"- Camera {camera}: `{image}` — robot pose "
            f"({float(row['robot_x']):.2f}, {float(row['robot_y']):.2f}) m, "
            f"yaw {float(row['robot_yaw']):.3f} rad"
        )
    (OUT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    make_five_camera_showcase()
    make_robot_detail()
    copy_camera_frames()
    write_source_note()
    print(f"wrote warehouse showcase to {OUT}")


if __name__ == "__main__":
    main()
