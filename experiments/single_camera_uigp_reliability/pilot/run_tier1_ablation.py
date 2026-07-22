#!/usr/bin/env python3
"""Tier-1 rejection-strategy ablation — does health-awareness add value?

Isolates the outlier-REJECTION strategy under a calibration drift, everything else
(route, affine calibration, max-jump limiter, planner) held constant:

  B0 raw_fuse : NIS gate OFF (threshold 0). Fuse the faulted camera -> the danger a
                per-frame gate was hiding.
  B1 nis_gate : fixed per-frame NIS gate at 9.21 (honest_v1). The strong baseline.
  B2 health   : NIS gate OFF, but the health-gated measurement filter drops the camera
                once the integrated-innovation health monitor latches DEGRADED
                (planner rides odom). Tests whether health catches what per-frame NIS misses.

Hypothesis: at MODERATE drift (0.5 m, which sneaks under the per-frame gate and left
B1 stuck at 0.66 m belief), B2 <= B1 << B0 belief error — i.e. integrated-innovation
health detection beats a per-frame NIS gate for slow/consistent drift.

Each run is one isolated, collision-safe real-Gazebo drive (run_tier1_pilot.run_one),
paired by seed across the three conditions. Run DETACHED; poll <out>/STATUS.md.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tier1_pilot import REPO, run_one, wait_machine_free, machine_busy  # noqa: E402

SEVERITIES = [0.5, 0.8]            # moderate (sneaks under gate) + large (gate rejects)
SEEDS = [0, 1, 2]
ONSET = 15.0
CONDITIONS = {  # label -> (nis_threshold, health_gate)
    "B0_raw_fuse": (0.0, False),
    "B1_nis_gate": (9.21, False),
    "B2_health":   (0.0, True),
}
FIELDS = ["idx", "severity", "onset", "condition", "seed", "reached_degraded",
          "degraded_latency_s", "min_health", "crashed", "goal_reached", "outcome",
          "mean_belief_error_gt_m", "runner_rc"]


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in rows:
            w.writerow({k: m.get(k) for k in FIELDS})


def summarize(rows):
    def fnum(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None
    sevs = sorted({m["severity"] for m in rows})
    lines = ["severity | condition    | n | belief err (mean of valid) | goal | collide | invalid | detected"]
    lines.append("-" * 92)
    for s in sevs:
        for cond in CONDITIONS:
            ms = [m for m in rows if m["severity"] == s and m["condition"] == cond]
            if not ms:
                continue
            n = len(ms)
            bes = [fnum(m.get("mean_belief_error_gt_m")) for m in ms]
            bes = [b for b in bes if b is not None]
            be = f"{sum(bes)/len(bes):.2f} m (n={len(bes)})" if bes else "  n/a  "
            goal = sum(1 for m in ms if m.get("goal_reached") is True)
            coll = sum(1 for m in ms if m.get("crashed") is True)
            inval = sum(1 for m in ms if (m.get("outcome") or "") == "infra_invalid" or m.get("aborted"))
            det = sum(1 for m in ms if m.get("reached_degraded"))
            lines.append(f"{s:>7} | {cond:<12} | {n} | {be:>26} | {goal:^4} | {coll:^7} | {inval:^7} | {det}/{n}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs" / "studies" /
                    "single_camera_uigp_reliability" / f"tier1_ablation_{int(time.time())}"))
    ap.add_argument("--task", default="route_apron_to_a3_mid")
    ap.add_argument("--condition", default="C2")
    ap.add_argument("--drift-ramp-s", type=float, default=15.0)
    ap.add_argument("--run-timeout-after-first-cmd-s", type=float, default=150.0)
    ap.add_argument("--domain-base", type=int, default=191)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / "STATUS.md"
    csv_path = out / "ablation.csv"
    if machine_busy():
        status.write_text("# Ablation ABORTED — machine not free at start\n")
        return 2

    runs = [(sev, seed, cond) for sev in SEVERITIES for seed in SEEDS for cond in CONDITIONS]
    started = time.time()
    rows = []
    status.write_text(f"# Tier-1 rejection-strategy ablation RUNNING\n\nstarted {time.ctime(started)} · "
                      f"{len(runs)} runs (B0/B1/B2 x {len(SEVERITIES)} sev x {len(SEEDS)} seeds)\n")
    for i, (sev, seed, cond) in enumerate(runs):
        wait_machine_free(timeout_s=60.0)
        nis_thr, hgate = CONDITIONS[cond]
        cell_out = out / f"run_{i:02d}_sev{sev}_{cond}_s{seed}"
        m = run_one(cell_out, domain=args.domain_base + i, partition=f"tier1abl{i}",
                    task=args.task, condition=args.condition, fault="calib_drift",
                    fault_after_s=ONSET, lookat_drift=f"{sev},{sev},0.0",
                    drift_ramp_s=args.drift_ramp_s,
                    run_timeout_after_first_cmd_s=args.run_timeout_after_first_cmd_s,
                    hard_timeout_s=400.0, seed=seed, nis_threshold=nis_thr, health_gate=hgate,
                    require_free=False)
        m.update(idx=i, severity=sev, onset=ONSET, condition=cond)
        rows.append(m)
        write_csv(csv_path, rows)
        status.write_text(
            f"# Tier-1 ablation RUNNING — {i+1}/{len(runs)} done\n\nelapsed {(time.time()-started)/60:.1f} min\n\n"
            f"last: sev={sev} {cond} s{seed} -> belief={m.get('mean_belief_error_gt_m')} "
            f"outcome={m.get('outcome')} detected={m.get('reached_degraded')}\n\n```\n{summarize(rows)}\n```\n")
        time.sleep(8)

    final = (f"# Tier-1 rejection-strategy ablation — DONE\n\nout: `{out}`\n\n{len(runs)} runs · "
             f"{(time.time()-started)/60:.1f} min\n\n```\n{summarize(rows)}\n```\n\n"
             f"B0 raw-fuse (no gate) · B1 fixed NIS gate 9.21 (honest_v1) · B2 health-gated rejection.\n"
             f"Claim to check: at 0.5 m (moderate) B2 <= B1 << B0 belief error. Full grid: `ablation.csv`.\n")
    status.write_text(final)
    (out / "RESULTS.md").write_text(final)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
