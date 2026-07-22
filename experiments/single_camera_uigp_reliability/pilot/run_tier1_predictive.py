#!/usr/bin/env python3
"""Tier-1 predictive-response comparison — does responding EARLIER cut the transient?

The ablation showed health-gated rejection (reject on DEGRADED) wins on belief accuracy
but still crashes in the 3-11 s detection-latency transient (before rejection engages).
This tests whether rejecting EARLIER closes that gap:

  B1  fixed NIS gate 9.21 (honest_v1)                      reference baseline
  B2  health-gated, reject on DEGRADED (debounced, latest) the ablation winner on accuracy
  B2p health-gated, reject on h < 0.5 (EARLY / predictive)  fires before the debounced DEGRADED

Benign drifts keep h > 0.55 (envelope), so the 0.5 early threshold does not false-trigger.
More seeds (n=5) to firm up the collision axis. Everything else constant. Paired by seed.

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

SEVERITIES = [0.5, 0.8]
SEEDS = [0, 1, 2, 3, 4]
ONSET = 15.0
# label -> (nis_threshold, health_gate, health_below)
CONDITIONS = {
    "B1_nis_gate": (9.21, False, 0.0),
    "B2_degraded": (0.0, True, 0.0),     # reject on debounced DEGRADED
    "B2p_early":   (0.0, True, 0.5),     # reject early on h < 0.5
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
    lines = ["severity | condition    | n | belief err (valid) | goal | collide | detected"]
    lines.append("-" * 82)
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
            det = sum(1 for m in ms if m.get("reached_degraded"))
            lines.append(f"{s:>7} | {cond:<12} | {n} | {be:>18} | {goal:^4} | {coll:^7} | {det}/{n}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs" / "studies" /
                    "single_camera_uigp_reliability" / f"tier1_predictive_{int(time.time())}"))
    ap.add_argument("--task", default="route_apron_to_a3_mid")
    ap.add_argument("--condition", default="C2")
    ap.add_argument("--drift-ramp-s", type=float, default=15.0)
    ap.add_argument("--run-timeout-after-first-cmd-s", type=float, default=150.0)
    ap.add_argument("--domain-base", type=int, default=191)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / "STATUS.md"
    csv_path = out / "predictive.csv"
    if machine_busy():
        status.write_text("# Predictive ABORTED — machine not free at start\n")
        return 2

    runs = [(sev, seed, cond) for sev in SEVERITIES for seed in SEEDS for cond in CONDITIONS]
    started = time.time()
    rows = []
    status.write_text(f"# Tier-1 predictive-response RUNNING\n\nstarted {time.ctime(started)} · "
                      f"{len(runs)} runs (B1/B2/B2p x {len(SEVERITIES)} sev x {len(SEEDS)} seeds)\n")
    for i, (sev, seed, cond) in enumerate(runs):
        wait_machine_free(timeout_s=60.0)
        nis_thr, hgate, hbelow = CONDITIONS[cond]
        cell_out = out / f"run_{i:02d}_sev{sev}_{cond}_s{seed}"
        m = run_one(cell_out, domain=args.domain_base + i, partition=f"tier1pred{i}",
                    task=args.task, condition=args.condition, fault="calib_drift",
                    fault_after_s=ONSET, lookat_drift=f"{sev},{sev},0.0",
                    drift_ramp_s=args.drift_ramp_s,
                    run_timeout_after_first_cmd_s=args.run_timeout_after_first_cmd_s,
                    hard_timeout_s=400.0, seed=seed, nis_threshold=nis_thr,
                    health_gate=hgate, health_gate_health_below=hbelow, require_free=False)
        m.update(idx=i, severity=sev, onset=ONSET, condition=cond)
        rows.append(m)
        write_csv(csv_path, rows)
        status.write_text(
            f"# Tier-1 predictive RUNNING — {i+1}/{len(runs)} done\n\nelapsed {(time.time()-started)/60:.1f} min\n\n"
            f"last: sev={sev} {cond} s{seed} -> belief={m.get('mean_belief_error_gt_m')} "
            f"crashed={m.get('crashed')} outcome={m.get('outcome')}\n\n```\n{summarize(rows)}\n```\n")
        time.sleep(8)

    final = (f"# Tier-1 predictive-response comparison — DONE\n\nout: `{out}`\n\n{len(runs)} runs · "
             f"{(time.time()-started)/60:.1f} min\n\n```\n{summarize(rows)}\n```\n\n"
             f"B1 fixed NIS gate · B2 reject-on-DEGRADED · B2p reject-early-on-h<0.5.\n"
             f"Question: does B2p cut the transient collisions vs B2 while keeping the accuracy win? "
             f"Full grid: `predictive.csv`.\n")
    status.write_text(final)
    (out / "RESULTS.md").write_text(final)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
