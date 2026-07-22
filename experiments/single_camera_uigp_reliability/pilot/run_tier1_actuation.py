#!/usr/bin/env python3
"""Tier-1 N3 actuation comparison — does acting on the health signal buy safety?

Re-runs the fault cells where the detection-half loop DETECTED the drift but the
robot (with no actuation) still collided/got stuck, now comparing:
  N0  safe_stop=False : health monitored + logged, robot keeps driving on the fault
  N3  safe_stop=True  : safe-degradation gate latches a STOP on DEGRADED
PAIRED by seed (same seed for N0 and N3 of a cell → only the gate differs). Each run
is one isolated, collision-safe real-Gazebo drive (run_tier1_pilot.run_one).

Headline metric: collision rate N0 vs N3 on the bad cells; plus a benign control
(undetected drift) to confirm N3 does NOT over-trigger (still reaches goal).

Run DETACHED; poll <out>/STATUS.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tier1_pilot import REPO, run_one, wait_machine_free, machine_busy  # noqa: E402

# (severity aim-slide m, onset s, label): the two envelope collision cells + a benign control
BAD_CELLS = [(0.5, 15.0, "collide@0.5/15"), (0.8, 30.0, "collide@0.8/30")]
CONTROL_CELLS = [(0.1, 15.0, "benign@0.1/15")]
BAD_SEEDS = [0, 1, 2]
CONTROL_SEEDS = [0, 1]
FIELDS = ["idx", "label", "severity", "onset", "mode", "seed", "reached_degraded",
          "degraded_latency_s", "min_health", "crashed", "outcome", "goal_reached",
          "mean_belief_error_gt_m", "runner_rc"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in rows:
            w.writerow({k: m.get(k) for k in FIELDS})


def summarize(rows):
    """Per-cell N0 vs N3 collision / safe-stop / goal counts."""
    cells = {}
    for m in rows:
        key = m["label"]
        c = cells.setdefault(key, {"N0": [], "N3": []})
        c["N3" if m["mode"] == "N3" else "N0"].append(m)
    lines = ["cell                | mode | n | collide | goal | other(stop/stuck) | detected"]
    lines.append("-" * 78)
    for label, modes in cells.items():
        for mode in ("N0", "N3"):
            ms = modes[mode]
            if not ms:
                continue
            n = len(ms)
            coll = sum(1 for m in ms if m.get("crashed") is True)
            goal = sum(1 for m in ms if m.get("goal_reached") is True)
            other = n - coll - goal
            det = sum(1 for m in ms if m.get("reached_degraded"))
            lines.append(f"{label:<19} | {mode:<4} | {n} | {coll:^7} | {goal:^4} | {other:^17} | {det}/{n}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs" / "studies" /
                    "single_camera_uigp_reliability" / f"tier1_actuation_{int(time.time())}"))
    ap.add_argument("--task", default="route_apron_to_a3_mid")
    ap.add_argument("--condition", default="C2")
    ap.add_argument("--drift-ramp-s", type=float, default=15.0)
    ap.add_argument("--run-timeout-after-first-cmd-s", type=float, default=90.0)
    ap.add_argument("--domain-base", type=int, default=191)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / "STATUS.md"
    csv_path = out / "actuation.csv"
    if machine_busy():
        status.write_text("# Actuation ABORTED — machine not free at start\n")
        return 2

    # build the run list: bad cells x 3 seeds x {N0,N3}, control x 2 seeds x {N0,N3}
    runs = []
    for (sev, onset, label) in BAD_CELLS:
        for seed in BAD_SEEDS:
            for mode in ("N0", "N3"):
                runs.append((sev, onset, label, seed, mode))
    for (sev, onset, label) in CONTROL_CELLS:
        for seed in CONTROL_SEEDS:
            for mode in ("N0", "N3"):
                runs.append((sev, onset, label, seed, mode))

    started = time.time()
    rows = []
    status.write_text(f"# Tier-1 N3 actuation RUNNING\n\nstarted {time.ctime(started)} · "
                      f"{len(runs)} runs (paired N0/N3 by seed)\n")
    for i, (sev, onset, label, seed, mode) in enumerate(runs):
        wait_machine_free(timeout_s=60.0)
        cell_out = out / f"run_{i:02d}_{label.replace('/','-').replace('@','_')}_{mode}_s{seed}"
        m = run_one(cell_out, domain=args.domain_base + i, partition=f"tier1act{i}",
                    task=args.task, condition=args.condition, fault="calib_drift",
                    fault_after_s=onset, lookat_drift=f"{sev},{sev},0.0",
                    drift_ramp_s=args.drift_ramp_s,
                    run_timeout_after_first_cmd_s=args.run_timeout_after_first_cmd_s,
                    hard_timeout_s=400.0, seed=seed, safe_stop=(mode == "N3"),
                    require_free=False)
        m.update(idx=i, label=label, severity=sev, onset=onset, mode=mode)
        rows.append(m)
        write_csv(csv_path, rows)
        status.write_text(
            f"# Tier-1 N3 actuation RUNNING — {i+1}/{len(runs)} done\n\n"
            f"elapsed {(time.time()-started)/60:.1f} min\n\n"
            f"last: {label} {mode} seed{seed} -> crashed={m.get('crashed')} "
            f"outcome={m.get('outcome')} detected={m.get('reached_degraded')}\n\n"
            f"```\n{summarize(rows)}\n```\n")
        time.sleep(8)

    final = (f"# Tier-1 N3 actuation — DONE\n\nout: `{out}`\n\n{len(runs)} runs · "
             f"{(time.time()-started)/60:.1f} min\n\n```\n{summarize(rows)}\n```\n\n"
             f"N0 = keep driving on the faulted camera; N3 = safe-stop on DEGRADED (latched).\n"
             f"Headline = collision count N0 vs N3 on the bad cells; benign control checks N3 "
             f"does not over-trigger. Full grid: `actuation.csv`.\n")
    status.write_text(final)
    (out / "RESULTS.md").write_text(final)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
