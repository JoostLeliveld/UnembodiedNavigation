"""
Dry-run preview of GP and YOLO capture positions for a given world.

Replicates the position-computation logic from capture_visibility_samples.py
and capture_simseg_dataset.py without launching Gazebo. Reads world_profiles.yaml
and tasks.yaml to draw the warehouse layout, then overlays accepted/rejected
sample positions for both pipelines.

Usage:
    python3 scripts/visibility_comparison/preview_capture_positions.py \
        --world warehouse_aws.world.sdf \
        --out /tmp/capture_preview.png --show
"""

import argparse
import pathlib
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PROFILES_DEFAULT = _REPO_ROOT / "src/experiments/config/world_profiles.yaml"
_TASKS_DEFAULT = _REPO_ROOT / "src/experiments/config/tasks.yaml"


def _in_any_region(x: float, y: float, regions: list, shrink_m: float) -> bool:
    for r in regions:
        if (float(r['xmin']) + shrink_m <= x <= float(r['xmax']) - shrink_m and
                float(r['ymin']) + shrink_m <= y <= float(r['ymax']) - shrink_m):
            return True
    return False


def _compute_positions(vis: dict, wall_margin: float, nx: int, ny: int,
                       traversable: list, shrink_m: float):
    xmin = float(vis['visibility_map_min_x']) + wall_margin
    xmax = float(vis['visibility_map_max_x']) - wall_margin
    ymin = float(vis['visibility_map_min_y']) + wall_margin
    ymax = float(vis['visibility_map_max_y']) - wall_margin

    xs = np.linspace(xmin, xmax, max(nx, 1))
    ys = np.linspace(ymin, ymax, max(ny, 1))

    accepted, rejected = [], []
    for y in ys:
        for x in xs:
            pt = (float(x), float(y))
            if not traversable or _in_any_region(x, y, traversable, shrink_m):
                accepted.append(pt)
            else:
                rejected.append(pt)
    return accepted, rejected


def _draw_panel(ax, title: str, accepted: list, rejected: list,
                regions: list, occlusions: list, task_arrows: list,
                world_bounds: dict, yaw_samples: int, dot_color: str):
    xlo = float(world_bounds['visibility_map_min_x'])
    xhi = float(world_bounds['visibility_map_max_x'])
    ylo = float(world_bounds['visibility_map_min_y'])
    yhi = float(world_bounds['visibility_map_max_y'])

    # Wall boundary
    ax.set_xlim(xlo - 0.2, xhi + 0.2)
    ax.set_ylim(ylo - 0.6, yhi + 0.2)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    wall_rect = mpatches.FancyBboxPatch(
        (xlo, ylo), xhi - xlo, yhi - ylo,
        boxstyle='square,pad=0', linewidth=1.5,
        edgecolor='black', facecolor='#f5f5f5', zorder=0)
    ax.add_patch(wall_rect)

    # Regions
    for r in regions:
        rtype = str(r.get('type', ''))
        rx = float(r['xmin'])
        ry = float(r['ymin'])
        rw = float(r['xmax']) - rx
        rh = float(r['ymax']) - ry
        if rtype == 'traversable':
            fc = '#d4edda'
            ec = '#6aab7a'
            alpha = 0.45
            zorder = 1
        elif 'non_driveable' in rtype or 'staging' in rtype:
            fc = '#f8d7da'
            ec = '#c0545a'
            alpha = 0.55
            zorder = 1
        else:
            continue
        patch = mpatches.Rectangle((rx, ry), rw, rh,
                                    facecolor=fc, edgecolor=ec,
                                    alpha=alpha, linewidth=0.6, zorder=zorder)
        ax.add_patch(patch)

    # Rack outlines (traversable aisles already colored; draw rack bodies separately)
    rack_regions = [r for r in regions
                    if 'rack' in str(r.get('name', '')) and 'aisle' not in str(r.get('name', ''))
                    and 'staging' not in str(r.get('type', ''))]
    for r in rack_regions:
        rx, ry = float(r['xmin']), float(r['ymin'])
        rw = float(r['xmax']) - rx
        rh = float(r['ymax']) - ry
        patch = mpatches.Rectangle((rx, ry), rw, rh,
                                    facecolor='#cccccc', edgecolor='#888888',
                                    alpha=0.7, linewidth=0.8, zorder=2)
        ax.add_patch(patch)

    # R4 high stack (first traversability=driveable_centerline occlusion)
    for occ in occlusions:
        if 'r4' in str(occ.get('name', '')).lower() or 'stack' in str(occ.get('cause', '')).lower():
            reg = occ.get('approx_region', {})
            if reg:
                rx, ry = float(reg['xmin']), float(reg['ymin'])
                rw = float(reg['xmax']) - rx
                rh = float(reg['ymax']) - ry
                patch = mpatches.Rectangle((rx, ry), rw, rh,
                                            facecolor='#8b4513', edgecolor='#5c2d0a',
                                            alpha=0.35, linewidth=1.0, zorder=3,
                                            label='R4 shadow zone')
                ax.add_patch(patch)
            break

    # Rejected grid positions (faint background)
    if rejected:
        rx_arr = [p[0] for p in rejected]
        ry_arr = [p[1] for p in rejected]
        ax.scatter(rx_arr, ry_arr, s=4, color='#aaaaaa', alpha=0.3, zorder=4,
                   linewidths=0)

    # Accepted positions
    if accepted:
        ax_arr = [p[0] for p in accepted]
        ay_arr = [p[1] for p in accepted]
        ax.scatter(ax_arr, ay_arr, s=12, color=dot_color, alpha=0.85, zorder=5,
                   linewidths=0)

    # Task arrows
    colors = {'B1': '#e03030', 'B2': '#2060c8', 'B3': '#9030c0', 'S': '#c07010'}
    for name, sx, sy, gx, gy in task_arrows:
        c = colors.get(name, '#333333')
        ax.annotate('', xy=(gx, gy), xytext=(sx, sy),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.4),
                    zorder=6)
        ax.scatter([sx], [sy], marker='s', s=28, color=c, zorder=7)
        ax.scatter([gx], [gy], marker='*', s=60, color=c, zorder=7)
        mid_x = (sx + gx) / 2 + 0.12
        mid_y = (sy + gy) / 2
        ax.text(mid_x, mid_y, name, fontsize=6, color=c, zorder=8,
                ha='left', va='center')

    # Camera marker
    cam_x = 0.0
    cam_y = ylo
    ax.plot(cam_x, cam_y, marker='^', markersize=9, color='#007bff',
            markeredgecolor='#003d80', zorder=8, label='camera (south wall)')
    ax.text(cam_x + 0.15, cam_y + 0.08, 'camera\n(south wall)',
            fontsize=5.5, color='#007bff', va='bottom', zorder=8)

    n_pos = len(accepted)
    n_cap = n_pos * yaw_samples
    ax.set_title(f'{title}\n({n_pos} positions × {yaw_samples} yaws = {n_cap} captures)',
                 fontsize=9)


def main():
    ap = argparse.ArgumentParser(description='Preview capture positions without Gazebo.')
    ap.add_argument('--world', default='warehouse_aws.world.sdf')
    ap.add_argument('--sample-nx', type=int, default=24)
    ap.add_argument('--sample-ny', type=int, default=20)
    ap.add_argument('--yaw-samples', type=int, default=4)
    ap.add_argument('--gp-wall-margin', type=float, default=0.45)
    ap.add_argument('--yolo-wall-margin', type=float, default=0.65)
    ap.add_argument('--region-shrink-m', type=float, default=0.05)
    ap.add_argument('--out', default='/tmp/capture_preview.png')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--profiles', default=str(_PROFILES_DEFAULT))
    ap.add_argument('--tasks', default=str(_TASKS_DEFAULT))
    args = ap.parse_args()

    with open(args.profiles) as f:
        profiles = yaml.safe_load(f)
    with open(args.tasks) as f:
        tasks_all = yaml.safe_load(f)

    world_cfg = profiles.get('worlds', {}).get(args.world)
    if world_cfg is None:
        print(f'ERROR: world {args.world!r} not found in {args.profiles}', file=sys.stderr)
        sys.exit(1)

    vis = world_cfg.get('visibility_defaults', {})
    regions = list(world_cfg.get('known_2d_regions') or [])
    occlusions = list(world_cfg.get('occlusion_annotations') or [])

    traversable = [r for r in regions if str(r.get('type', '')) == 'traversable']

    gp_acc, gp_rej = _compute_positions(
        vis, args.gp_wall_margin, args.sample_nx, args.sample_ny,
        traversable, args.region_shrink_m)
    yolo_acc, yolo_rej = _compute_positions(
        vis, args.yolo_wall_margin, args.sample_nx, args.sample_ny,
        traversable, args.region_shrink_m)

    # Task arrows
    world_tasks = tasks_all.get('tasks', {}).get(args.world, [])
    task_arrows = []
    for t in world_tasks:
        name = t.get('name', '')
        label = None
        if name == 'B1_apron_a4_to_uppermid_a3':
            label = 'B1'
        elif 'sanity' in name.lower() and 'aws' in name.lower():
            label = 'S'
        if label and label not in [a[0] for a in task_arrows]:
            s = t.get('start', {})
            g = t.get('goal', {})
            task_arrows.append((label,
                                 float(s.get('x', 0)), float(s.get('y', 0)),
                                 float(g.get('x', 0)), float(g.get('y', 0))))

    print(f'GP:   {len(gp_acc)} positions × {args.yaw_samples} yaws = '
          f'{len(gp_acc) * args.yaw_samples} captures  '
          f'({args.sample_nx}×{args.sample_ny} grid, wall_margin={args.gp_wall_margin}m)')
    print(f'YOLO: {len(yolo_acc)} positions × {args.yaw_samples} yaws = '
          f'{len(yolo_acc) * args.yaw_samples} captures  '
          f'({args.sample_nx}×{args.sample_ny} grid, wall_margin={args.yolo_wall_margin}m)')

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(f'AWS Warehouse — Capture Positions ({args.world})', fontsize=11, y=1.01)

    _draw_panel(axes[0], f'GP visibility capture\n(wall_margin={args.gp_wall_margin}m)',
                gp_acc, gp_rej, regions, occlusions, task_arrows, vis,
                args.yaw_samples, '#1a9e8a')
    _draw_panel(axes[1], f'YOLO dataset capture\n(wall_margin={args.yolo_wall_margin}m)',
                yolo_acc, yolo_rej, regions, occlusions, task_arrows, vis,
                args.yaw_samples, '#e07a10')

    legend_elements = [
        mpatches.Patch(facecolor='#d4edda', edgecolor='#6aab7a', label='traversable region'),
        mpatches.Patch(facecolor='#f8d7da', edgecolor='#c0545a', label='non-driveable staging'),
        mpatches.Patch(facecolor='#8b4513', alpha=0.35, label='R4 shadow zone'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1a9e8a',
                   markersize=7, label='GP sample position'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#e07a10',
                   markersize=7, label='YOLO sample position'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#aaaaaa',
                   markersize=5, alpha=0.5, label='rejected (outside traversable)'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#e03030',
                   markersize=6, label='B1 start/goal'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2060c8',
                   markersize=6, label='B2 start/goal'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#9030c0',
                   markersize=6, label='B3 start/goal'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#c07010',
                   markersize=6, label='S start/goal'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               fontsize=7.5, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)

    plt.tight_layout()
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
