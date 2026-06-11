#!/usr/bin/env python3
"""Create the paper localization-pathway figure.

The figure is intentionally mechanism-focused. It shows that the external
camera supplies an image-space detection, that the runtime selects the bottom
pixel of the YOLO box/mask as the ground-contact point, and that homography
projection yields a planar `(x, y)` update. Heading is shown as a separate
odometry branch, because the paper-facing runtime does not use YOLO heading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.transforms import Affine2D
import numpy as np
import yaml


REPO = Path(__file__).resolve().parents[2]
THESIS = REPO.parent / "thesis-report"

# Current-world (camera z=4.8/y=-5.5) capture frame at true (-4.61,-1.67), robot visible;
# oracle bottom pixel (159.2,345.8) matches world_to_pixel for the profile camera. The
# previous aws_simseg_v2 training image was removed during repo cleanup.
DEFAULT_IMAGE = REPO / "logs/visibility_comparison/aws_capture_v7b_col461/images/000012_xy0003_h00.jpg"
DEFAULT_MODEL = REPO / "logs/perception_models/aws_yolo_simseg_v2/model.pt"
DEFAULT_GP = REPO / "logs/visibility_comparison/aws_gp_v7b/yolo_score_raw_gp.npz"
DEFAULT_PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
DEFAULT_OUT = THESIS / "figures/localization_pathway.pdf"
DEFAULT_PREVIEW = REPO / "logs/paper_figures/localization_pathway.png"

sys.path.insert(0, str(REPO / "src/unav_common"))
sys.path.insert(0, str(REPO / "src/perception"))
from perception.core.yolo_selection import select_best_detection, target_class_ids  # noqa: E402
from unav_common.camera_model import ObliqueCameraModel  # noqa: E402


PINK = "#e6194b"
CYAN = "#00bfc4"
BLUE = "#2171b5"
GREEN = "#7fbf7b"
WARM_BG = "#eeece8"
TEXT = "#222222"
GRAY = "#777777"
STROKE = [pe.withStroke(linewidth=4, foreground="white")]

# Image-plane parallelogram in schematic coordinates.
IP_BL = np.array([2.35, 4.55])
IP_BR = np.array([6.55, 4.55])
IP_TL = np.array([3.65, 7.55])
IP_TR = np.array([7.85, 7.55])
AFF = Affine2D.from_values(
    IP_BR[0] - IP_BL[0],
    IP_BR[1] - IP_BL[1],
    IP_TL[0] - IP_BL[0],
    IP_TL[1] - IP_BL[1],
    IP_BL[0],
    IP_BL[1],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--gp", type=Path, default=DEFAULT_GP)
    parser.add_argument("--world", default="warehouse_aws.world.sdf")
    parser.add_argument("--world-profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview", type=Path, default=DEFAULT_PREVIEW)
    parser.add_argument("--conf", type=float, default=0.10)
    return parser.parse_args()


def detect_robot(img_bgr: np.ndarray, model_path: Path, conf: float) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    result = model.predict(
        source=img_bgr,
        imgsz=640,
        conf=conf,
        iou=0.45,
        verbose=False,
    )[0]
    target_ids = target_class_ids(getattr(model, "names", {}), "robot", -1)
    selected = select_best_detection(
        result,
        target_ids=target_ids,
        confidence_threshold=conf,
        use_masks=True,
        mask_min_area=12.0,
        mask_bottom_band_px=3.0,
    )
    if not selected:
        raise RuntimeError("No robot detection selected for localization-pathway figure")
    return selected


def traversable_rects(profile_path: Path, world: str) -> list[dict]:
    raw = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    return [
        item
        for item in raw["worlds"][world].get("known_2d_regions", [])
        if item.get("type") == "traversable"
    ]


def to_affine(points: np.ndarray) -> np.ndarray:
    return AFF.transform(np.asarray(points, dtype=float))


def draw_camera_icon(ax, center=(1.0, 9.82)) -> np.ndarray:
    cam_c = np.asarray(center, dtype=float)
    ax.add_patch(
        FancyBboxPatch(
            tuple(cam_c + [-0.46, -0.30]),
            0.92,
            0.60,
            boxstyle="round,pad=0.05",
            facecolor="#3a3a3a",
            edgecolor="black",
            lw=1.35,
            zorder=5,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            tuple(cam_c + [-0.12, 0.31]),
            0.24,
            0.17,
            boxstyle="round,pad=0.025",
            facecolor="#555555",
            edgecolor="black",
            lw=0.9,
            zorder=5,
        )
    )
    lens = cam_c + [0.55, 0.0]
    ax.add_patch(Circle(tuple(lens), 0.23, facecolor="#555555", edgecolor="black", lw=1.25, zorder=6))
    ax.add_patch(Circle(tuple(lens), 0.13, facecolor="#85a7d8", edgecolor="none", zorder=7))
    ax.text(cam_c[0], cam_c[1] + 0.62, "fixed camera", ha="center", va="bottom", fontsize=11, color="#444444")
    return lens


def draw_left_panel(ax, image_rgb: np.ndarray, selected: dict) -> tuple[float, float, np.ndarray | None]:
    h, w = image_rgb.shape[:2]
    ax.set_xlim(0, 9.7)
    ax.set_ylim(3.25, 10.55)
    ax.axis("off")
    ax.set_title("(a) YOLO image observation", fontsize=13, fontweight="bold", pad=6)

    lens = draw_camera_icon(ax)
    for corner in (IP_BL, IP_BR, IP_TR, IP_TL):
        ax.plot([lens[0], corner[0]], [lens[1], corner[1]], "--", color="#8a8a8a", lw=1.1, zorder=1)

    trans = AFF + ax.transData
    im = ax.imshow(
        np.flipud(image_rgb),
        extent=[0, 1, 0, 1],
        origin="lower",
        aspect="auto",
        interpolation="bilinear",
        transform=trans,
        zorder=2,
    )
    clip_poly = Polygon([IP_BL, IP_BR, IP_TR, IP_TL], closed=True, transform=ax.transData)
    im.set_clip_path(clip_poly)
    ax.add_patch(Polygon([IP_BL, IP_BR, IP_TR, IP_TL], closed=True, fill=False, edgecolor=CYAN, lw=2.0, zorder=4))
    ax.text(((IP_BL + IP_BR) / 2)[0], IP_BL[1] - 0.28, "camera image plane", ha="center", va="top", fontsize=10, style="italic", color="#555555")

    bbox = selected.get("bbox_xyxy")
    if bbox is not None:
        x0, y0, x1, y1 = [float(v) for v in np.asarray(bbox).reshape(-1)[:4]]
        bbox_norm = np.array(
            [
                [x0 / w, 1.0 - y0 / h],
                [x1 / w, 1.0 - y0 / h],
                [x1 / w, 1.0 - y1 / h],
                [x0 / w, 1.0 - y1 / h],
            ],
            dtype=float,
        )
        bbox_s = to_affine(bbox_norm)
        ax.add_patch(Polygon(bbox_s, closed=True, fill=False, edgecolor=CYAN, lw=2.2, zorder=6))
        ax.text(
            bbox_s[:, 0].mean() - 0.25,
            bbox_s[:, 1].max() + 0.34,
            "YOLO\nbox/mask",
            ha="center",
            va="bottom",
            fontsize=9.4,
            color=CYAN,
            fontweight="bold",
            path_effects=STROKE,
            zorder=7,
        )

    u = float(selected["selected_u"])
    v = float(selected["selected_v"])
    selected_source = str(selected.get("selected_pixel_source", "bbox_bottom"))
    point_s = to_affine(np.array([[u / w, 1.0 - v / h]], dtype=float))[0]
    ax.plot(*point_s, "*", color=PINK, ms=20, mec="black", mew=0.8, zorder=8)
    ax.annotate(
        f"selected bottom pixel\n({selected_source})",
        xy=point_s,
        xytext=(point_s[0] + 0.72, point_s[1] + 1.55),
        color=PINK,
        fontsize=9.8,
        fontweight="bold",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": PINK, "lw": 1.6},
        path_effects=STROKE,
        zorder=9,
    )
    ax.text(point_s[0] + 0.62, point_s[1] - 0.32, r"$(u_{\rm pix},v_{\rm pix})$", color=PINK, fontsize=11.5, fontweight="bold", path_effects=STROKE, zorder=9)
    return u, v, bbox


def draw_middle_panel(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) projection model", fontsize=13, fontweight="bold", pad=6)

    ax.add_patch(FancyArrowPatch((0.08, 0.68), (0.92, 0.68), arrowstyle="-|>", mutation_scale=28, lw=3.8, color=BLUE))
    ax.text(0.50, 0.78, "ground-plane\nhomography", ha="center", va="bottom", fontsize=12.5, color=TEXT, fontweight="bold")
    ax.text(
        0.50,
        0.51,
        r"$[\tilde{x},\tilde{y},\tilde{w}]^\top = H^{-1}[u_{\rm pix},v_{\rm pix},1]^\top$",
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=TEXT,
    )
    ax.text(
        0.50,
        0.39,
        r"$(\hat{x},\hat{y})=(\tilde{x}/\tilde{w},\,\tilde{y}/\tilde{w})$",
        ha="center",
        va="bottom",
        fontsize=10.2,
        color=TEXT,
    )

    # (Removed the "separate heading branch / odometry fallback" box: heading is supplied
    # by odometry in the paper-facing runtime — it is the architecture, not a fallback.)


def draw_ground_panel(ax, profile_path: Path, world: str, gp_path: Path, x_hat: float, y_hat: float) -> None:
    with np.load(gp_path, allow_pickle=False) as data:
        cam_pos = tuple(float(v) for v in data["camera_pos"][:3])
        look_at = tuple(float(v) for v in data["look_at"][:3])
        img_width = int(np.asarray(data["img_width"]).reshape(-1)[0])
        img_height = int(np.asarray(data["img_height"]).reshape(-1)[0])
        fov_h_rad = float(np.asarray(data["fov_h_rad"]).reshape(-1)[0])
    del img_width, img_height, fov_h_rad
    cam_xy = (float(cam_pos[0]), float(cam_pos[1]))

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor(WARM_BG)
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)
        spine.set_color("black")
    ax.set_title("(c) planner-facing update", fontsize=13, fontweight="bold", pad=6)

    # Plot-only: the profile's mid_cross_aisle is stored as one full-width band, but the
    # R0 (leftmost) shelf is CONTINUOUS (no mid gap), so it blocks that band over its
    # footprint. Subtract the R0 footprint when drawing so no driveable "cross" is shown
    # over R0. Only mid_cross_aisle overlaps R0's footprint; other lanes are unchanged.
    _R0 = (-4.325, -3.775, -0.82, 4.25)  # leftmost continuous shelf footprint

    def _rects_minus_r0(rx0, rx1, ry0, ry1):
        bx0, bx1, by0, by1 = _R0
        oy0, oy1 = max(ry0, by0), min(ry1, by1)
        ox0, ox1 = max(rx0, bx0), min(rx1, bx1)
        if oy1 <= oy0 or ox1 <= ox0:
            yield (rx0, rx1, ry0, ry1)
            return
        if ry0 < oy0:
            yield (rx0, rx1, ry0, oy0)
        if ry1 > oy1:
            yield (rx0, rx1, oy1, ry1)
        if rx0 < bx0:
            yield (rx0, bx0, oy0, oy1)
        if rx1 > bx1:
            yield (bx1, rx1, oy0, oy1)

    for rect in traversable_rects(profile_path, world):
        for sx0, sx1, sy0, sy1 in _rects_minus_r0(
            rect["xmin"], rect["xmax"], rect["ymin"], rect["ymax"]
        ):
            ax.add_patch(
                Rectangle(
                    (sx0, sy0),
                    sx1 - sx0,
                    sy1 - sy0,
                    facecolor="#d8ead5",
                    edgecolor=GREEN,
                    lw=1.0,
                    zorder=1,
                )
            )

    ax.plot(cam_xy[0], cam_xy[1], "s", ms=9, color="#444444", zorder=5)
    ax.text(cam_xy[0] + 0.16, cam_xy[1] - 0.22, "camera", fontsize=10.5, color="#444444", va="top")
    ax.plot([cam_xy[0], x_hat], [cam_xy[1], y_hat], ":", color="#8a8a8a", lw=1.6, zorder=2)
    ax.plot(x_hat, y_hat, "*", color=PINK, ms=25, mec="black", mew=0.9, zorder=6)
    ax.annotate(
        r"camera update: $(\hat{x},\hat{y})$",
        xy=(x_hat, y_hat),
        xytext=(x_hat - 2.75, y_hat + 0.88),
        color=PINK,
        fontsize=10.8,
        fontweight="bold",
        ha="left",
        arrowprops={"arrowstyle": "->", "color": PINK, "lw": 1.6},
    )
    # Show heading as an odometry-derived orientation at the same state.
    ax.add_patch(FancyArrowPatch((x_hat, y_hat), (x_hat + 0.62, y_hat), arrowstyle="-|>", mutation_scale=15, lw=1.8, color="#444444", zorder=7))
    ax.text(x_hat - 0.15, y_hat - 0.62, r"$\hat{\theta}$ from odometry", ha="right", va="center", fontsize=10.5, color="#444444")

    origin = (-5.15, -5.05)
    ax.annotate("", xy=(origin[0] + 0.75, origin[1]), xytext=origin, arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.1})
    ax.annotate("", xy=(origin[0], origin[1] + 0.75), xytext=origin, arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.1})
    ax.text(origin[0] + 0.92, origin[1] - 0.03, "$x$", color=GRAY, fontsize=9, va="top")
    ax.text(origin[0] - 0.05, origin[1] + 0.90, "$y$", color=GRAY, fontsize=9, ha="right")

    ax.text(
        0.03,
        0.96,
        "camera contributes position only\nGP changes camera $(x,y)$ covariance",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.6,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.92},
    )
    ax.set_xlim(-5.8, 5.8)
    ax.set_ylim(-5.6, 5.2)


def main() -> int:
    args = parse_args()
    img_bgr = cv2.imread(str(args.image))
    if img_bgr is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    selected = detect_robot(img_bgr, args.model, args.conf)
    u = float(selected["selected_u"])
    v = float(selected["selected_v"])
    with np.load(args.gp, allow_pickle=False) as data:
        cam = ObliqueCameraModel(
            cam_pos=tuple(float(x) for x in data["camera_pos"][:3]),
            look_at=tuple(float(x) for x in data["look_at"][:3]),
            img_width=int(np.asarray(data["img_width"]).reshape(-1)[0]),
            img_height=int(np.asarray(data["img_height"]).reshape(-1)[0]),
            fov_h_rad=float(np.asarray(data["fov_h_rad"]).reshape(-1)[0]),
        )
    x_hat, y_hat = cam.pixel_to_world(u, v)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(14.2, 5.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.12, 2.25])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_mid = fig.add_subplot(gs[0, 1])
    ax_ground = fig.add_subplot(gs[0, 2])

    draw_left_panel(ax_img, image_rgb, selected)
    draw_middle_panel(ax_mid)
    draw_ground_panel(ax_ground, args.world_profile, args.world, args.gp, x_hat, y_hat)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.preview, dpi=220, bbox_inches="tight")
    plt.close(fig)

    caption = (
        "Image-space YOLO detection is reduced to a selected bottom pixel, "
        "which is projected to a planar ground point with the calibrated "
        "homography/camera model. The external camera contributes (x,y) "
        "position updates; heading is supplied by odometry. The GP "
        "modulates the planner-facing camera (x,y) covariance and does not "
        "directly estimate heading."
    )
    caption_path = args.preview.with_name("localization_pathway_caption.txt")
    caption_path.write_text(caption + "\n", encoding="utf-8")
    print(f"selected_pixel=({u:.1f}, {v:.1f}) source={selected.get('selected_pixel_source', 'unknown')}")
    print(f"projected_xy=({x_hat:.2f}, {y_hat:.2f})")
    print(f"wrote {args.out}")
    print(f"wrote {args.preview}")
    print(f"wrote {caption_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
