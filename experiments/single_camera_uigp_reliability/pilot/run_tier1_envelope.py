#!/usr/bin/env python3
"""Tier-1 safe-operating-ENVELOPE sweep (contribution C1).

Sweeps the calibration-drift fault over (health severity) x (onset location) and
maps, per cell, whether/when the health monitor fires DEGRADED and what happens to
the robot (belief error, outcome) on REAL Gazebo perception. This is the C1
safe-operating-envelope characterization: safety & detectability as a function of
fault severity and where on the route (camera-coverage geometry) it strikes.

Each cell is one isolated, collision-safe run via run_tier1_pilot.run_one (own
ROS_DOMAIN_ID + partition, single run => no runner broad pkill, own-pgroup
teardown). Cells run sequentially with a machine-free wait + cleanup delay between.

Severity = camera aim-point slide (m) at full drift (re-projected through real
geometry => pose-dependent world error). Onset = seconds after first detection
(=> different route progress / camera coverage when the fault strikes).

Run DETACHED and poll <out>/STATUS.md:
  setsid python3 run_tier1_envelope.py --out <dir> > <dir>/env.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_tier1_pilot import REPO, run_one, wait_machine_free, machine_busy  # noqa: E402

SEVERITIES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8]   # aim-slide metres at full drift
ONSETS = [15.0, 30.0]                          # s after first detection (route progress)
FIELDS = ["idx", "severity", "onset", "reached_degraded", "degraded_latency_s",
          "min_health", "max_nis_ewma", "max_bias", "baseline_false_alarms",
          "healthy_frames", "degraded_frames", "n_frames",
          "goal_reached", "crashed", "outcome", "mean_belief_error_gt_m", "runner_rc", "aborted"]


def write_csv(path: Path, results):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for m in results:
            w.writerow({k: m.get(k) for k in FIELDS})


def surface_tables(results):
    """ASCII surface: rows=severity, cols=onset. One table for detection latency,
    one for outcome/belief-error."""
    sevs = sorted({m["severity"] for m in results})
    onsets = sorted({m["onset"] for m in results})
    by = {(m["severity"], m["onset"]): m for m in results}

    def cell_det(m):
        if m is None or m.get("aborted"):
            return "  --  "
        if not m.get("reached_degraded"):
            return " none "
        return f"{m['degraded_latency_s']:>4}s "

    def cell_out(m):
        if m is None or m.get("aborted"):
            return "   --    "
        be = m.get("mean_belief_error_gt_m")
        be = f"{be:.2f}" if isinstance(be, (int, float)) else "?"
        oc = (m.get("outcome") or "?")[:5]
        return f"{oc:>5}/{be:>4}"

    def render(title, fn):
        hdr = "sev\\onset | " + " | ".join(f"{o:>7.0f}s" for o in onsets)
        rows = [title, hdr, "-" * len(hdr)]
        for s in sevs:
            rows.append(f"{s:>8.2f} | " + " | ".join(f"{fn(by.get((s, o))):>8}" for o in onsets))
        return "\n".join(rows)

    return (render("DETECTION latency (onset->DEGRADED); 'none'=undetected", cell_det)
            + "\n\n" + render("OUTCOME/belief-err(m)", cell_out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "logs" / "studies" /
                    "single_camera_uigp_reliability" / f"tier1_envelope_{int(time.time())}"))
    ap.add_argument("--task", default="route_apron_to_a3_mid")
    ap.add_argument("--condition", default="C2")
    ap.add_argument("--drift-ramp-s", type=float, default=15.0)
    ap.add_argument("--domain-base", type=int, default=191)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = out / "STATUS.md"
    csv_path = out / "envelope.csv"

    if machine_busy():
        status.write_text("# Envelope ABORTED — machine not free at start\n")
        return 2

    cells = [(sev, onset) for onset in ONSETS for sev in SEVERITIES]
    started = time.time()
    results = []
    status.write_text(f"# Tier-1 envelope RUNNING\n\nstarted {time.ctime(started)}\n"
                      f"grid: {len(SEVERITIES)} severities x {len(ONSETS)} onsets = {len(cells)} cells\n")

    for i, (sev, onset) in enumerate(cells):
        wait_machine_free(timeout_s=60.0)
        cell_out = out / f"cell_{i:02d}_sev{sev}_on{int(onset)}"
        m = run_one(cell_out, domain=args.domain_base + i, partition=f"tier1env{i}",
                    task=args.task, condition=args.condition, fault="calib_drift",
                    fault_after_s=onset, lookat_drift=f"{sev},{sev},0.0",
                    drift_ramp_s=args.drift_ramp_s, run_timeout_after_first_cmd_s=150.0,
                    hard_timeout_s=600.0, require_free=False)
        m.update(idx=i, severity=sev, onset=onset)
        results.append(m)
        write_csv(csv_path, results)
        elapsed = time.time() - started
        status.write_text(
            f"# Tier-1 envelope RUNNING — {i+1}/{len(cells)} cells done\n\n"
            f"elapsed {elapsed/60:.1f} min · started {time.ctime(started)}\n\n"
            f"last cell: sev={sev} onset={onset} -> reached_degraded={m.get('reached_degraded')} "
            f"latency={m.get('degraded_latency_s')} outcome={m.get('outcome')} "
            f"belief_err={m.get('mean_belief_error_gt_m')}\n\n```\n{surface_tables(results)}\n```\n")
        time.sleep(8)  # cleanup delay between cells

    final = (f"# Tier-1 safe-operating-envelope — DONE\n\n"
             f"out: `{out}`\n\n{len(cells)} cells · {(time.time()-started)/60:.1f} min\n\n"
             f"```\n{surface_tables(results)}\n```\n\n"
             f"Full grid: `envelope.csv`. Each cell dir has health_trace.csv + campaign/.../experiment.csv.\n")
    status.write_text(final)
    (out / "RESULTS.md").write_text(final)
    print(final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
