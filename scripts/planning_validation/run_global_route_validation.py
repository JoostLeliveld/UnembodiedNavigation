#!/usr/bin/env python3
"""Offline validation of the corrected global route objective (no Gazebo).

Runs the full-route selector over the thesis tasks under a grid of observability
conditions and both objectives (legacy and corrected), and writes a new, versioned
artifact directory. Existing runs, logs, routes and manifests are never touched
and never overwritten (ground rules 3 and 4): the script refuses to write into a
directory that already exists.

Grid
----
tasks       : the blind-corridor diagnostic + the four principal thesis tasks
              (plus the sanity crossing), from src/experiments/config/tasks.yaml
footprints  : `spec_0.80x0.55`   -- the body the planner is required to fit
              `deployed_burger`  -- the TurtleBot3 burger body actually simulated
conditions  : q in {unity, commissioned} x R in {constant_global, commissioned}
objectives  : `legacy_efe` (pre-correction accounting) and `corrected_v2`
gate sets   : `full`     -- every hard gate on, including body-inside-driveable
              `deployment` -- body-vs-obstacle and path-inside-driveable hard,
                            body-inside-driveable reported but not vetoing

Outputs (in the new artifact directory)
---------------------------------------
  candidates.csv          every candidate's safety status, length, minimum body
                          clearance, q/R localization cost, total cost and
                          selected/not-selected status
  selection_comparison.csv  legacy vs corrected selection per task/condition
  breakeven.csv           the localization weight at which each ranking flips
  summary.md              the readable report, including which Gazebo runs need
                          repeating
  results.json            the full decomposition for every candidate
  manifest.json           provenance: config, geometry, GP artifact, weights,
                          gate and execution signatures, git revision

Run:
  python3 scripts/planning_validation/run_global_route_validation.py
  python3 scripts/planning_validation/run_global_route_validation.py --tag rerun2
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_planner_setup import (  # noqa: E402
    DEFAULT_WORLD,
    ObservabilityCondition,
    build_global_planner,
    build_safety_model,
    lane_graph_candidates,
    load_campaign_config,
    load_tasks,
)
from planning.core.global_route_selector import (  # noqa: E402
    ExecutionModelConfig,
    GlobalRouteSelector,
    RouteObjectiveConfig,
)
from planning.core.route_safety import RobotFootprint, SafetyGateConfig  # noqa: E402

BLIND_CORRIDOR_TASK = "occlusion_transit_a4"
PRINCIPAL_TASKS = (
    "route_apron_to_a3_mid",
    "route_apron_to_a2_mid",
    "route_west_to_a1_upper",
    "control_west_to_a1_low",
)
TASKS = (BLIND_CORRIDOR_TASK,) + PRINCIPAL_TASKS + ("sanity_visible_apron_crossing",)

FOOTPRINTS = {
    # The body the corrected planner is required to fit.
    "spec_0.80x0.55": RobotFootprint(0.80, 0.55),
    # The body the Gazebo campaign actually drives (TurtleBot3 burger: 0.140 m
    # base box, 0.178 m across the wheels), for which the shipped
    # robot_collision_radius_m = 0.125 is the matching inflation radius.
    "deployed_burger": RobotFootprint(0.140, 0.178),
}

CONDITIONS = [
    ObservabilityCondition("unity", "constant_global"),
    ObservabilityCondition("unity", "commissioned"),
    ObservabilityCondition("commissioned", "constant_global"),
    ObservabilityCondition("commissioned", "commissioned"),
]

# The one declared exchange rate between observability and travel: seconds of
# extra travel that are worth avoiding one nat of excess measurement entropy
# sustained for one second. Declared here, not fitted (ground rule 5); the
# break-even table reports how far each decision sits from this value.
LOCALIZATION_WEIGHT = 1.0

CORRECTED = RouteObjectiveConfig(
    mode="corrected_v2",
    travel_time_weight=1.0,
    route_length_weight=0.0,
    localization_weight=LOCALIZATION_WEIGHT,
    goal_risk_weight=0.0,
    obstacle_weight=0.0,
    discount_gamma=1.0,
)
LEGACY = RouteObjectiveConfig(
    mode="legacy_efe",
    travel_time_weight=0.0,
    route_length_weight=0.0,
    localization_weight=0.0,
    goal_risk_weight=1.0,
    obstacle_weight=1.0,
    discount_gamma=0.995,
    risk_uses_reference_R=False,
)
OBJECTIVES = {"legacy_efe": LEGACY, "corrected_v2": CORRECTED}

GATE_SETS = {
    "full": dict(require_body_inside_driveable=True),
    "deployment": dict(require_body_inside_driveable=False),
}


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _fmt(value, spec="{:.3f}") -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return spec.format(value)
    return str(value)


def run_grid(cfg: dict, tasks: dict) -> list[dict]:
    rows: list[dict] = []
    # Safety verdicts are condition- and objective-independent, so one safety
    # model per (footprint, gate set) is reused across the whole grid.
    safety_models = {
        (fp_name, gate_name): build_safety_model(
            cfg,
            footprint=footprint,
            gates=SafetyGateConfig(
                goal_radius_m=float(cfg.get("goal_success_radius", 0.25)), **gate_kwargs
            ),
        )
        for fp_name, footprint in FOOTPRINTS.items()
        for gate_name, gate_kwargs in GATE_SETS.items()
    }

    r_ref = float(cfg["r_visible_uv"])
    R_reference = np.diag([r_ref ** 2, r_ref ** 2]).astype(float)
    execution = ExecutionModelConfig(dt=0.1, v_max=float(cfg["v_max"]), w_max=1.0)
    sigma0 = float(cfg.get("init_belief_sigma_xy", 0.05))
    sigma0_theta = float(cfg.get("init_belief_sigma_theta", 0.05))
    S0 = np.diag([sigma0 ** 2, sigma0 ** 2, sigma0_theta ** 2]).astype(float)

    for task_name in TASKS:
        task = tasks[task_name]
        start = np.array(
            [float(task["start"]["x"]), float(task["start"]["y"]),
             float(task["start"].get("yaw", 0.0))],
            dtype=float,
        )
        goal = np.array([float(task["goal"]["x"]), float(task["goal"]["y"])], dtype=float)
        candidates = lane_graph_candidates(cfg, start[:2], goal)
        if not candidates:
            print(f"[warn] {task_name}: lane-graph produced no candidates; skipped")
            continue

        for condition in CONDITIONS:
            planner = build_global_planner(cfg, condition, world_path=DEFAULT_WORLD)
            for objective_name, objective in OBJECTIVES.items():
                for (fp_name, gate_name), safety in safety_models.items():
                    selector = GlobalRouteSelector(
                        planner,
                        safety,
                        execution=execution,
                        objective=objective,
                        R_reference=R_reference,
                    )
                    result = selector.select(candidates, start, S0, goal)
                    for ev in result.evaluations:
                        row = ev.as_row()
                        row.update(
                            task=task_name,
                            condition=condition.label,
                            q_mode=condition.q_mode,
                            r_mode=condition.r_mode,
                            objective=objective_name,
                            footprint=fp_name,
                            gate_set=gate_name,
                            failed_gates=",".join(ev.safety.failed_gates),
                            turn_angle_total_rad=ev.turn_angle_total_rad,
                            terminal_belief_sigma_xy_m=ev.terminal_belief_sigma_xy_m,
                            ambiguity_reference_nat_s=ev.ambiguity_reference_nat_s,
                            goal_risk_cost_reference_R=ev.goal_risk_cost_reference_R,
                            goal_risk_cost_plan_R=ev.goal_risk_cost_plan_R,
                        )
                        rows.append(row)
            print(f"  done: {task_name:30s} {condition.label}")
    return rows


def breakeven_table(rows: list[dict]) -> list[dict]:
    """The localization weight at which each corrected ranking would flip.

    Reported so that the sensitivity of every route decision to the one declared
    exchange rate is visible, rather than the weight being tuned until a route
    flips.
    """
    out: list[dict] = []
    keys = sorted({
        (r["task"], r["condition"], r["footprint"], r["gate_set"])
        for r in rows if r["objective"] == "corrected_v2"
    })
    for task, condition, footprint, gate_set in keys:
        group = [
            r for r in rows
            if r["objective"] == "corrected_v2" and r["task"] == task
            and r["condition"] == condition and r["footprint"] == footprint
            and r["gate_set"] == gate_set and r["safe"]
        ]
        if len(group) < 2:
            out.append({
                "task": task, "condition": condition, "footprint": footprint,
                "gate_set": gate_set, "n_safe": len(group),
                "shortest": group[0]["route"] if group else "-",
                "most_observable": "-", "break_even_localization_weight": math.nan,
                "selected_at_declared_weight": group[0]["route"] if group else "-",
                "note": "fewer than two safe candidates",
            })
            continue
        shortest = min(group, key=lambda r: r["travel_time_s"])
        other = min(
            (r for r in group if r["route"] != shortest["route"]),
            key=lambda r: r["localization_cost_nat_s"],
        )
        d_travel = other["travel_time_s"] - shortest["travel_time_s"]
        d_loc = shortest["localization_cost_nat_s"] - other["localization_cost_nat_s"]
        if d_loc > 1e-12:
            break_even = d_travel / d_loc
            note = "detour justified above this weight"
        else:
            break_even = math.inf
            note = "the longer route is never more observable; no weight can justify it"
        selected = next((r["route"] for r in group if r["selected"]), "-")
        out.append({
            "task": task, "condition": condition, "footprint": footprint,
            "gate_set": gate_set, "n_safe": len(group),
            "shortest": shortest["route"], "most_observable": other["route"],
            "delta_travel_s": d_travel, "delta_localization_nat_s": d_loc,
            "break_even_localization_weight": break_even,
            "declared_localization_weight": LOCALIZATION_WEIGHT,
            "selected_at_declared_weight": selected, "note": note,
        })
    return out


def selection_comparison(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    keys = sorted({(r["task"], r["condition"], r["footprint"], r["gate_set"]) for r in rows})
    for task, condition, footprint, gate_set in keys:
        def pick(objective: str):
            sel = [
                r["route"] for r in rows
                if r["objective"] == objective and r["task"] == task
                and r["condition"] == condition and r["footprint"] == footprint
                and r["gate_set"] == gate_set and r["selected"]
            ]
            return sel[0] if sel else "<none>"

        legacy = pick("legacy_efe")
        corrected = pick("corrected_v2")
        out.append({
            "task": task, "condition": condition, "footprint": footprint,
            "gate_set": gate_set, "legacy_selected": legacy,
            "corrected_selected": corrected, "selection_changed": legacy != corrected,
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path: Path, rows: list[dict], comparison: list[dict],
                  breakeven: list[dict], cfg: dict) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Corrected global route objective -- offline validation")
    add("")
    add(f"Generated: {_dt.datetime.now().isoformat(timespec='seconds')}")
    add("")
    add("Objective under test:")
    add("")
    add("```")
    add("minimize   travel_time")
    add("         + localization_weight * INTEGRAL c_loc(q(t), R(t)) dt")
    add("subject to collision-free, inside the driveable region,")
    add("           the complete body fits (straights, corners, in-place turns),")
    add("           kinematically feasible, reaches the goal")
    add("")
    add("c_loc = max( 0.5*log( det R_plan(q,R) / det R_ref ), 0 )   [nats, >= 0]")
    add("```")
    add("")
    add(f"Declared localization weight: {LOCALIZATION_WEIGHT} s per nat-second "
        f"(not fitted; see breakeven.csv for the sensitivity of every decision).")
    add(f"Reference measurement quality R_ref: r = {cfg['r_visible_uv']} px.")
    add("")

    # --- per-task candidate tables ---------------------------------------
    add("## Candidate tables (corrected objective)")
    add("")
    for gate_set in GATE_SETS:
        for footprint in FOOTPRINTS:
            subset = [
                r for r in rows
                if r["objective"] == "corrected_v2" and r["footprint"] == footprint
                and r["gate_set"] == gate_set
            ]
            if not subset:
                continue
            add(f"### footprint = {footprint}, gate set = {gate_set}")
            add("")
            add("| task | condition | route | safety | len [m] | T [s] | min body clr [m] "
                "| q mean | q min | loc cost [s] | total | selected |")
            add("|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in subset:
                add(
                    f"| {r['task']} | {r['condition']} | {r['route']} | {r['safety_status']} "
                    f"| {_fmt(r['route_length_m'], '{:.2f}')} | {_fmt(r['travel_time_s'], '{:.1f}')} "
                    f"| {_fmt(r['min_body_clearance_m'], '{:+.3f}')} "
                    f"| {_fmt(r['mean_q'])} | {_fmt(r['min_q'])} "
                    f"| {_fmt(r['localization_cost'], '{:.2f}')} "
                    f"| {_fmt(r['total_cost'], '{:.2f}')} | {_fmt(r['selected'])} |"
                )
            add("")

    # --- old vs corrected -------------------------------------------------
    add("## Old vs corrected selection")
    add("")
    add("| task | condition | footprint | gate set | legacy | corrected | changed |")
    add("|---|---|---|---|---|---|---|")
    for c in comparison:
        add(f"| {c['task']} | {c['condition']} | {c['footprint']} | {c['gate_set']} "
            f"| {c['legacy_selected']} | {c['corrected_selected']} "
            f"| {'YES' if c['selection_changed'] else 'no'} |")
    add("")

    # --- why the legacy objective chose what it chose ----------------------
    add("## What actually drove the legacy choice")
    add("")
    add("The legacy total is dominated on some tasks by the soft no-go penalty "
        "(a safety-shaping term), not by an observability trade-off. Removing that "
        "term shows which route the legacy *observability* accounting preferred on "
        "its own.")
    add("")
    add("| task | condition | legacy (total) | legacy (excl. no-go) | no-go share of total |")
    add("|---|---|---|---|---|")
    legacy_safe = [
        r for r in rows
        if r["objective"] == "legacy_efe" and r["footprint"] == "deployed_burger"
        and r["gate_set"] == "full" and r["safe"]
    ]
    keys = sorted({(r["task"], r["condition"]) for r in legacy_safe})
    for task, condition in keys:
        group = [r for r in legacy_safe if r["task"] == task and r["condition"] == condition]
        with_nogo = min(group, key=lambda r: r["total_cost"])["route"]
        without = min(group, key=lambda r: r["total_cost_excl_obstacle"])["route"]
        share = max(
            abs(r["obstacle_cost"]) / max(abs(r["total_cost"]), 1e-9) for r in group
        )
        add(f"| {task} | {condition} | {with_nogo} | {without} | {_fmt(share, '{:.3f}')} |")
    add("")

    # --- break-even -------------------------------------------------------
    add("## Break-even localization weight")
    add("")
    add("| task | condition | footprint | gate set | shortest | alternative | dT [s] "
        "| dLoc [nat.s] | break-even weight | selected |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for b in breakeven:
        add(f"| {b['task']} | {b['condition']} | {b['footprint']} | {b['gate_set']} "
            f"| {b['shortest']} | {b.get('most_observable', '-')} "
            f"| {_fmt(b.get('delta_travel_s'), '{:.1f}')} "
            f"| {_fmt(b.get('delta_localization_nat_s'), '{:.2f}')} "
            f"| {_fmt(b.get('break_even_localization_weight'), '{:.3f}')} "
            f"| {b.get('selected_at_declared_weight', '-')} |")
    add("")

    # --- validation gates -------------------------------------------------
    add("## Validation gates")
    add("")
    unity_rows = [
        r for r in rows
        if r["objective"] == "corrected_v2" and r["q_mode"] == "unity" and r["safe"]
    ]
    unity_ok = True
    for key in sorted({(r["task"], r["condition"], r["footprint"], r["gate_set"])
                       for r in unity_rows}):
        group = [r for r in unity_rows if (r["task"], r["condition"], r["footprint"],
                                           r["gate_set"]) == key]
        if not group:
            continue
        shortest = min(group, key=lambda r: r["travel_time_s"])["route"]
        selected = [r["route"] for r in group if r["selected"]]
        if selected and selected[0] != shortest:
            unity_ok = False
            add(f"- **FAIL** q=1 did not select the shortest safe route for {key}: "
                f"selected {selected[0]}, shortest {shortest}")
    add(f"- q=1 selects the shortest safe route: **{'PASS' if unity_ok else 'FAIL'}**")

    loc_ok = all(
        r["localization_cost"] >= -1e-12
        for r in rows if r["objective"] == "corrected_v2"
    )
    add(f"- localization cost is never negative (no reward for duration): "
        f"**{'PASS' if loc_ok else 'FAIL'}**")

    zero_ok = all(
        abs(r["localization_cost"]) < 1e-9
        for r in rows
        if r["objective"] == "corrected_v2" and r["q_mode"] == "unity"
        and r["r_mode"] == "constant_global"
    )
    add(f"- q=1 with constant/global R gives exactly zero localization cost: "
        f"**{'PASS' if zero_ok else 'FAIL'}**")

    unsafe_selected = [r for r in rows if r["selected"] and not r["safe"]]
    add(f"- no unsafe route was ever selected: "
        f"**{'PASS' if not unsafe_selected else 'FAIL'}**")

    no_selection = [c for c in comparison if c["corrected_selected"] == "<none>"]
    if no_selection:
        add("")
        add(f"- {len(no_selection)} (task, condition, footprint, gate set) combinations "
            "had NO safe candidate, so no route was selected. See the candidate table "
            "for the failing gate.")
    add("")

    # --- Gazebo reruns ----------------------------------------------------
    add("## Gazebo runs that must be repeated")
    add("")
    changed = [
        c for c in comparison
        if c["selection_changed"] and c["footprint"] == "deployed_burger"
        and c["gate_set"] == "full"
    ]
    if changed:
        add("Route selection changed for the deployed footprint under the full gate "
            "set, so the corresponding simulator runs are invalidated:")
        add("")
        for c in changed:
            add(f"- `{c['task']}` / {c['condition']}: {c['legacy_selected']} -> "
                f"{c['corrected_selected']}")
    else:
        add("No route selection changed for the deployed footprint under the full "
            "gate set; on that evidence no previously recorded run is invalidated by "
            "the route choice itself.")
    add("")
    add("Independently of route choice, every run recorded before this change used "
        "the pre-correction objective, so any figure that quotes an EFE cost "
        "decomposition (risk / ambiguity split, total cost) must be regenerated: the "
        "terms are defined differently now.")
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None,
                        help="suffix for the artifact directory (default: today's date)")
    parser.add_argument("--out-root", default="paper_artifacts/planning",
                        help="root under which the versioned artifact directory is created")
    args = parser.parse_args()

    tag = args.tag or _dt.date.today().strftime("%Y%m%d")
    out_dir = REPO_ROOT / args.out_root / f"global_objective_v2_{tag}"
    if out_dir.exists():
        print(f"refusing to overwrite existing artifact directory: {out_dir}\n"
              f"pass --tag <something-new> to create a new version", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)

    cfg = load_campaign_config()
    tasks = load_tasks()
    print(f"writing to {out_dir}")
    rows = run_grid(cfg, tasks)
    comparison = selection_comparison(rows)
    breakeven = breakeven_table(rows)

    write_csv(out_dir / "candidates.csv", rows)
    write_csv(out_dir / "selection_comparison.csv", comparison)
    write_csv(out_dir / "breakeven.csv", breakeven)
    (out_dir / "results.json").write_text(
        json.dumps({"candidates": rows, "comparison": comparison, "breakeven": breakeven},
                   indent=2, default=str),
        encoding="utf-8",
    )
    write_summary(out_dir / "summary.md", rows, comparison, breakeven, cfg)

    gp_path = REPO_ROOT / cfg["gp_artifact"]
    manifest = {
        "generated": _dt.datetime.now().isoformat(timespec="seconds"),
        "git_revision": _git_revision(),
        "campaign_config": "scripts/visibility_comparison/warehouse_visibility_campaign.yaml",
        "tasks_yaml": "src/experiments/config/tasks.yaml",
        "world": str(DEFAULT_WORLD.relative_to(REPO_ROOT)),
        "gp_artifact": cfg["gp_artifact"],
        "gp_artifact_sha256": _sha256_file(gp_path),
        "driveable_geometry_sha256": hashlib.sha256(
            str(cfg["driveable_geometry_json"]).encode("utf-8")
        ).hexdigest(),
        "tasks": list(TASKS),
        "conditions": [c.describe() for c in CONDITIONS],
        "footprints": {k: {"length_m": v.length_m, "width_m": v.width_m,
                           "circumradius_m": v.circumradius_m}
                       for k, v in FOOTPRINTS.items()},
        "gate_sets": GATE_SETS,
        "objectives": {k: v.describe() for k, v in OBJECTIVES.items()},
        "localization_weight": LOCALIZATION_WEIGHT,
        "r_reference_uv": float(cfg["r_visible_uv"]),
        "execution_model": {"dt": 0.1, "v_max": float(cfg["v_max"]), "w_max": 1.0},
        "n_candidate_rows": len(rows),
        "numpy": np.__version__,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nwrote {len(rows)} candidate rows")
    print(f"  {out_dir/'candidates.csv'}")
    print(f"  {out_dir/'selection_comparison.csv'}")
    print(f"  {out_dir/'breakeven.csv'}")
    print(f"  {out_dir/'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
