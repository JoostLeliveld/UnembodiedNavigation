#!/usr/bin/env python3
"""Create the appendix YOLO training-clarification figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


REPO = Path(__file__).resolve().parents[2]
THESIS = REPO.parent / "thesis-report"
VAL_GRID = REPO / "paper_artifacts/perception/warehouse_yolo_detector_v1/val_batch0_pred.jpg"
OUT = THESIS / "figures/appendix/yolo_training_clarification.pdf"
PREVIEW = REPO / "paper_artifacts/figures/yolo_training_clarification.png"


def crop_grid_cell(image: Image.Image, row: int, col: int, rows: int = 4, cols: int = 4) -> Image.Image:
    w, h = image.size
    x0 = int(col * w / cols)
    x1 = int((col + 1) * w / cols)
    y0 = int(row * h / rows)
    y1 = int((row + 1) * h / rows)
    return image.crop((x0, y0, x1, y1))


def box(ax, xy: tuple[float, float], text: str, color: str, *, w: float = 1.62, h: float = 0.64) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.055,rounding_size=0.035",
            facecolor=color,
            edgecolor="#333333",
            linewidth=1.0,
        )
    )
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center", fontsize=9.2, linespacing=0.92)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.05,
            color="#555555",
        )
    )


def main() -> int:
    grid = Image.open(VAL_GRID).convert("RGB")
    examples = [crop_grid_cell(grid, 0, 0), crop_grid_cell(grid, 3, 3)]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(6.45, 3.55), constrained_layout=False)
    ax = fig.add_axes([0.02, 0.07, 0.46, 0.86])
    ax.set_xlim(0, 5.7)
    ax.set_ylim(0, 3.45)
    ax.axis("off")
    ax.text(2.55, 3.34, "(a) training source", ha="center", va="top", fontsize=11.5, fontweight="bold")

    box(ax, (0.10, 2.20), "RGB frames\n+ semantic\nlabels", "#fff0bd")
    box(ax, (2.05, 2.20), "YOLO-seg\ndataset\n852 images", "#dcefd7")
    box(ax, (3.92, 2.20), "YOLOv11n-seg\nfine-tune\n30 epochs", "#e8e2f0")
    box(ax, (2.05, 1.12), "runtime RGB\ninference", "#f3e8dc")
    box(ax, (3.92, 1.12), "bounding-box\nbottom centre\nfor projection", "#f5dce0")

    arrow(ax, (1.72, 2.52), (2.05, 2.52))
    arrow(ax, (3.67, 2.52), (3.92, 2.52))
    arrow(ax, (4.73, 2.20), (4.73, 1.76))
    arrow(ax, (3.67, 1.44), (3.92, 1.44))

    ax.text(
        0.18,
        0.28,
        "Semantic labels create masks offline;\nruntime detector uses RGB only.",
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    fig.text(0.72, 0.94, "(b) validation predictions", ha="center", va="top", fontsize=11.5, fontweight="bold")
    image_axes = [
        fig.add_axes([0.51, 0.53, 0.47, 0.34]),
        fig.add_axes([0.51, 0.12, 0.47, 0.34]),
    ]
    for idx, (iax, img) in enumerate(zip(image_axes, examples, strict=True), start=1):
        iax.imshow(img)
        iax.set_xticks([])
        iax.set_yticks([])
        for spine in iax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)
            spine.set_color("#777777")
        iax.text(
            0.02,
            0.96,
            f"b{idx}",
            transform=iax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    fig.savefig(PREVIEW, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"wrote {PREVIEW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
