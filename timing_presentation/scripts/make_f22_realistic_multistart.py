#!/usr/bin/env python3
"""F22: realistic neutral multistart comparison for the AWS pick task.

This figure replaces diagnostic route labels such as "A3 seed" with candidate
initializations that a warehouse local planner could reasonably get:

* hold position,
* local escape arcs from the current shelf-facing pose,
* shortest and alternate routes from the known 2D driveable floor.

The same candidates are offered to C1 and C2. Route choice is then the minimum
valid optimized rollout under each condition's objective.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml


ROOT = Path("/home/joostleliveld/Thesis")
REPO = ROOT / "UnembodiedNavigation"
FIG_DIR = REPO / "timing_presentation" / "figures"
SCRIPT_DIR = REPO / "scripts" / "visibility_comparison"
CONFIG = SCRIPT_DIR / "aws_smoke_config.yaml"
TASK = "B1_apron_a4_to_uppermid_a3"
WORLD = "warehouse_aws.world.sdf"

OUT_PNG = FIG_DIR / "F22_realistic_multistart_choice.png"
OUT_CSV = FIG_DIR / "F22_realistic_multistart_choice.csv"
OUT_MD = FIG_DIR / "F22_realistic_multistart_choice.md"

HORIZON = 80
V_MAX = 1.1
W_MAX = 1.0
AMBIGUITY_WEIGHT = 8.0
GOAL_PRIOR_FINAL = 4.0
OPT_MAXITER = 100
OPT_MAXFUN = 160

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "src" / "experiments"))
sys.path.insert(0, str(REPO / "src" / "planning"))
sys.path.insert(0, str(REPO / "src" / "unav_common"))

from efe_offline_lab import build_planner, load_setup, run_optimizer  # noqa: E402


plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _load_profile() -> dict:
    path = REPO / "src" / "experiments" / "config" / "world_profiles.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["worlds"][WORLD]


def _region_center(profile: dict, name: str) -> tuple[float, float]:
    for region in profile.get("known_2d_regions", []) or []:
        if region.get("name") == name:
            return (
                0.5 * (float(region["xmin"]) + float(region["xmax"])),
                0.5 * (float(region["ymin"]) + float(region["ymax"])),
            )
    raise KeyError(name)


def _controls_for_waypoints(planner, start_xy_yaw, waypoints) -> np.ndarray:
    horizon = int(planner.horizon)
    dt = float(planner.dt)
    v_max = float(planner.v_max)
    w_max = float(planner.w_max)
    m = np.asarray(start_xy_yaw, dtype=float).reshape(3).copy()
    controls = np.zeros((horizon, 2), dtype=float)
    waypoint_idx = 0
    waypoints = [np.asarray(wp, dtype=float).reshape(2) for wp in waypoints]

    for k in range(horizon):
        if waypoint_idx >= len(waypoints):
            break
        target = waypoints[waypoint_idx]
        d = target - m[:2]
        if float(np.linalg.norm(d)) < 0.18 and waypoint_idx < len(waypoints) - 1:
            waypoint_idx += 1
            target = waypoints[waypoint_idx]
            d = target - m[:2]

        desired_yaw = math.atan2(float(d[1]), float(d[0]))
        yaw_err = _wrap_pi(desired_yaw - float(m[2]))
        w = float(np.clip(yaw_err / max(dt, 1e-6), -w_max, w_max))
        v = 0.65 * v_max if abs(yaw_err) < 0.65 else 0.0
        controls[k] = [v, w]
        m, _ = planner.predict(m, np.eye(3) * 1e-6, controls[k])
    return controls


def _local_escape_controls(planner, turn_sign: float) -> np.ndarray:
    horizon = int(planner.horizon)
    controls = np.zeros((horizon, 2), dtype=float)
    # Rotate from shelf-facing east into a corridor direction, then move slowly.
    turn_steps = min(7, horizon)
    controls[:turn_steps, 1] = turn_sign * 0.75 * float(planner.w_max)
    if horizon > turn_steps:
        controls[turn_steps:, 0] = 0.45 * float(planner.v_max)
    return controls


def _candidate_initial_controls(planner, setup, profile: dict) -> dict[str, np.ndarray]:
    sx, sy, _ = setup.start_xy_yaw
    gx, gy = setup.goal_xy
    a3_x, _ = _region_center(profile, "rack_aisle_A3")
    a4_x, _ = _region_center(profile, "rack_aisle_A4")
    _, lower_y = _region_center(profile, "lower_main_aisle")
    _, mid_y = _region_center(profile, "mid_cross_aisle")

    return {
        "zero_hold": np.zeros((planner.horizon, 2), dtype=float),
        "local_left_escape": _local_escape_controls(planner, +1.0),
        "local_right_escape": _local_escape_controls(planner, -1.0),
        "floor_route_upper_cross": _controls_for_waypoints(
            planner,
            setup.start_xy_yaw,
            [(a4_x, mid_y), (a3_x, mid_y), (gx, gy)],
        ),
        "floor_route_lower_sweep": _controls_for_waypoints(
            planner,
            setup.start_xy_yaw,
            [(a4_x, lower_y), (a3_x, lower_y), (a3_x, gy), (gx, gy)],
        ),
    }


def _trace_controls(planner, start_xy_yaw, controls) -> np.ndarray:
    m = np.asarray(start_xy_yaw, dtype=float).reshape(3).copy()
    states = [m[:2].copy()]
    S = np.eye(3) * 1e-6
    for u in np.asarray(controls, dtype=float).reshape(-1, 2):
        m, S = planner.predict(m, S, u)
        states.append(m[:2].copy())
    return np.asarray(states, dtype=float)


def _draw_regions(ax, profile: dict) -> None:
    first_drive = True
    first_nondrive = True
    for r in profile.get("known_2d_regions", []) or []:
        kind = str(r.get("type", ""))
        if kind == "traversable":
            ax.add_patch(Rectangle(
                (r["xmin"], r["ymin"]),
                r["xmax"] - r["xmin"],
                r["ymax"] - r["ymin"],
                facecolor="#cae9c9",
                edgecolor="#6abf69",
                lw=0.5,
                alpha=0.55,
                zorder=0,
                label="known driveable floor" if first_drive else None,
            ))
            first_drive = False
        elif "non_driveable" in kind:
            ax.add_patch(Rectangle(
                (r["xmin"], r["ymin"]),
                r["xmax"] - r["xmin"],
                r["ymax"] - r["ymin"],
                facecolor="#efb0b0",
                edgecolor="#d94848",
                lw=0.7,
                alpha=0.42,
                zorder=1,
                label="non-driveable" if first_nondrive else None,
            ))
            first_nondrive = False


def _draw_task(ax, setup) -> None:
    sx, sy, yaw = setup.start_xy_yaw
    gx, gy = setup.goal_xy
    ax.plot(sx, sy, "o", color="#111827", ms=8, zorder=12, label="start")
    ax.arrow(
        sx, sy,
        0.35 * math.cos(yaw), 0.35 * math.sin(yaw),
        width=0.02,
        head_width=0.11,
        head_length=0.13,
        color="#111827",
        length_includes_head=True,
        zorder=13,
    )
    ax.plot(gx, gy, "*", color="#111827", ms=15, zorder=12, label="goal")


def _format_map(ax, setup, title: str) -> None:
    ax.set_title(title)
    ax.set_xlim(0.25, 4.05)
    ax.set_ylim(-2.75, 2.35)
    ax.set_aspect("equal")
    ax.grid(alpha=0.22)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    _draw_task(ax, setup)


def _row_from_plan(condition: str, candidate: str, plan) -> dict:
    states = np.asarray(plan.states, dtype=float)
    return {
        "condition": condition,
        "candidate": candidate,
        "selected": 0,
        "total_cost": float(plan.total_cost),
        "risk_cost": float(plan.risk_cost),
        "ambiguity_cost": float(plan.ambiguity_cost),
        "drive_barrier_cost": float(plan.obstacle_cost),
        "terminal_goal_distance_m": float(plan.terminal_goal_distance_pred),
        "path_length_m": float(np.linalg.norm(np.diff(states[:, :2], axis=0), axis=1).sum()) if len(states) > 1 else 0.0,
        "rollout_valid": int(bool(plan.rollout_valid)),
        "optimizer_success": int(bool(plan.optimizer_success)),
        "optimizer_status": int(plan.optimizer_status),
        "optimizer_nfev": int(plan.optimizer_nfev),
        "solve_time_ms": 1000.0 * float(plan.solve_time_s),
        "x_min": float(np.min(states[:, 0])),
        "x_max": float(np.max(states[:, 0])),
        "y_min": float(np.min(states[:, 1])),
        "y_max": float(np.max(states[:, 1])),
    }


def _best_candidate(rows: list[dict], plans: dict[str, object]) -> str:
    valid = [r for r in rows if r["rollout_valid"] and math.isfinite(r["total_cost"])]
    pool = valid if valid else [r for r in rows if math.isfinite(r["total_cost"])]
    if not pool:
        return ""
    return str(min(pool, key=lambda r: float(r["total_cost"]))["candidate"])


def main() -> int:
    profile = _load_profile()
    setups = {
        "C1 constant-R": load_setup(CONFIG, condition="C1", seed=1, task_override=TASK),
        "C2 visibility-aware": load_setup(CONFIG, condition="C2", seed=1, task_override=TASK),
    }

    all_rows: list[dict] = []
    all_plans: dict[str, dict[str, object]] = {}
    all_inits: dict[str, np.ndarray] | None = None
    for label, setup in setups.items():
        cfg = dict(setup.config)
        cfg.update({
            "horizon": HORIZON,
            "v_max": V_MAX,
            "w_max": W_MAX,
            "ambiguity_weight": AMBIGUITY_WEIGHT,
            "goal_prior_u_std_final": GOAL_PRIOR_FINAL,
            "goal_prior_v_std_final": GOAL_PRIOR_FINAL,
            "optimizer_maxiter": OPT_MAXITER,
            "optimizer_maxfun": OPT_MAXFUN,
            "optimizer_multistart": False,
        })
        planner = build_planner(
            cfg,
            planner_kind=setup.planner_kind,
            camera_params=setup.camera_params,
            visibility_artifact_path=str(setup.gp_path),
            visibility_geometry_json=setup.geometry_json,
            collision_geometry_json=setup.geometry_json,
            driveable_geometry_json=setup.driveable_geometry_json,
            seed=1,
            visibility_target_height_m=setup.gp["target_height"],
        )
        candidates = _candidate_initial_controls(planner, setup, profile)
        if all_inits is None:
            all_inits = dict(candidates)

        plans: dict[str, object] = {}
        rows: list[dict] = []
        for name, controls in candidates.items():
            try:
                plan = run_optimizer(
                    planner,
                    np.asarray(setup.start_xy_yaw, dtype=float),
                    setup.S0,
                    np.asarray(setup.goal_xy, dtype=float),
                    u_init=controls.reshape(-1),
                )
                plans[name] = plan
                rows.append(_row_from_plan(label, name, plan))
            except Exception as exc:
                rows.append({
                    "condition": label,
                    "candidate": name,
                    "selected": 0,
                    "error": repr(exc),
                    "total_cost": math.nan,
                    "risk_cost": math.nan,
                    "ambiguity_cost": math.nan,
                    "drive_barrier_cost": math.nan,
                    "terminal_goal_distance_m": math.nan,
                    "path_length_m": math.nan,
                    "rollout_valid": 0,
                    "optimizer_success": 0,
                    "optimizer_status": 999,
                    "optimizer_nfev": 0,
                    "solve_time_ms": math.nan,
                    "x_min": math.nan,
                    "x_max": math.nan,
                    "y_min": math.nan,
                    "y_max": math.nan,
                })

        best = _best_candidate(rows, plans)
        for row in rows:
            row["selected"] = int(row["candidate"] == best)
        all_rows.extend(rows)
        all_plans[label] = plans

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(all_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    setup0 = next(iter(setups.values()))
    p_map = setup0.gp["P_plan"]
    xs = setup0.gp["xs"]
    ys = setup0.gp["ys"]
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))

    # Candidate initial guesses.
    ax = axes[0, 0]
    ax.imshow(p_map, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
              cmap="viridis", vmin=0, vmax=1.0, alpha=0.78, zorder=-2)
    _draw_regions(ax, profile)
    colors = {
        "zero_hold": "#6b7280",
        "local_left_escape": "#eab308",
        "local_right_escape": "#f97316",
        "floor_route_upper_cross": "#2563eb",
        "floor_route_lower_sweep": "#16a34a",
    }
    for name, controls in (all_inits or {}).items():
        tr = _trace_controls(setup0.planner, setup0.start_xy_yaw, controls)
        ax.plot(tr[:, 0], tr[:, 1], "--", lw=1.8, color=colors.get(name, "0.5"), label=name)
    _format_map(ax, setup0, "(a) Neutral candidate initializations")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.88)

    # Converged plans for C1/C2.
    for ax, (label, setup) in zip([axes[0, 1], axes[1, 0]], setups.items()):
        ax.imshow(p_map, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower",
                  cmap="viridis", vmin=0, vmax=1.0, alpha=0.78, zorder=-2)
        _draw_regions(ax, profile)
        rows = [r for r in all_rows if r["condition"] == label]
        selected = next((r["candidate"] for r in rows if int(r["selected"]) == 1), "")
        for name, plan in all_plans[label].items():
            states = np.asarray(plan.states, dtype=float)
            is_selected = name == selected
            ax.plot(
                states[:, 0],
                states[:, 1],
                "-",
                lw=3.2 if is_selected else 1.2,
                alpha=1.0 if is_selected else 0.35,
                color=colors.get(name, "0.5"),
                label=(f"SELECTED: {name}" if is_selected else name),
                zorder=6 if is_selected else 4,
            )
        _format_map(ax, setup, f"{'(b)' if label.startswith('C1') else '(c)'} {label}: optimized rollouts")
        ax.legend(loc="upper left", fontsize=7.5, framealpha=0.88)

    # Numeric cost table.
    ax = axes[1, 1]
    ax.axis("off")
    lines = [
        "F22: Same neutral candidates, different objective",
        f"H={HORIZON}, v_max={V_MAX}, 2σ driveable barrier",
        "",
        "Selected candidates:",
    ]
    for label in setups:
        rows = [r for r in all_rows if r["condition"] == label]
        sel = next((r for r in rows if int(r["selected"]) == 1), None)
        if sel:
            lines.append(
                f"{label}: {sel['candidate']} | "
                f"J={sel['total_cost']:.1f}, risk={sel['risk_cost']:.1f}, "
                f"amb={sel['ambiguity_cost']:.1f}, barrier={sel['drive_barrier_cost']:.1f}, "
                f"d={sel['terminal_goal_distance_m']:.2f} m"
            )
    lines.append("")
    lines.append("All candidate costs (J / terminal distance):")
    for label in setups:
        lines.append(label)
        rows = [r for r in all_rows if r["condition"] == label]
        for r in sorted(rows, key=lambda rr: (0 if rr["selected"] else 1, rr["candidate"])):
            mark = "*" if int(r["selected"]) else " "
            lines.append(
                f" {mark} {r['candidate']:<23} "
                f"J={r['total_cost']:7.1f}  d={r['terminal_goal_distance_m']:.2f}  "
                f"valid={int(r['rollout_valid'])}"
            )
        lines.append("")
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8.6)

    fig.suptitle(
        "F22 — Realistic neutral multistart: what does the initial planner choose?",
        fontsize=16,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160)

    selected_lines = []
    for label in setups:
        sel = next((r for r in all_rows if r["condition"] == label and int(r["selected"]) == 1), None)
        if sel:
            selected_lines.append(
                f"- {label}: `{sel['candidate']}` with J={sel['total_cost']:.1f}, "
                f"risk={sel['risk_cost']:.1f}, ambiguity={sel['ambiguity_cost']:.1f}, "
                f"barrier={sel['drive_barrier_cost']:.1f}, terminal d={sel['terminal_goal_distance_m']:.2f} m."
            )
    OUT_MD.write_text(
        "# F22 Realistic Neutral Multistart Choice\n\n"
        f"- figure: `{OUT_PNG}`\n"
        f"- samples: `{OUT_CSV}`\n\n"
        "## Setup\n\n"
        f"- task: `{TASK}`\n"
        "- start: `(3.20, -1.00, yaw=0.0)`, shelf-facing east\n"
        "- goal: `(1.00, 1.75)`\n"
        f"- horizon: `{HORIZON}`\n"
        "- candidates are condition-neutral optimizer initializations, not mission waypoints.\n"
        "- floor routes are condition-neutral route initializations through known driveable corridors only.\n"
        "- these labels are not claims about globally shortest graph paths.\n\n"
        "## Selected Initial Plans\n\n"
        + "\n".join(selected_lines)
        + "\n\n## Interpretation\n\n"
        "This figure is closer to the intended comparison than F21: both C1 and C2 receive the same "
        "realistic candidate set. The selected path is the best valid optimized rollout under each "
        "condition's objective. If C2 chooses a different route, that difference comes from planner-facing "
        "observation covariance and the ambiguity/risk terms, not from route scripting.\n",
        encoding="utf-8",
    )

    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")
    for line in selected_lines:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
