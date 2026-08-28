"""One consistent scoring pass over every configuration in the ladder.

Every error is scored against the truth at the instant the estimate describes, and the
correction count is the count of distinct detector rounds -- both through `aligned.py`,
the same loader `score.py` uses, so the ladder and the six-arm table cannot drift apart.

Two numbers here changed meaning on 2026-08-28 and are printed differently as a result:

* the stated 1-sigma is a MEDIAN, not a mean. Its mean is set by the few seconds of a
  drive spent in a correction outage, where the stated sigma reaches metres, so a mean of
  28 cm used to sit beside a median error of 2.75 cm with nothing marking them as
  descriptions of different parts of the run.
* NEES is shown as mean AND median, against their own targets (2.0 and 1.386). They
  disagree sharply here -- the belief is padded most of the time and overconfident in
  the tenth of the run following a correction gap -- and one summary number hides that.
"""
import glob
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aligned as A  # noqa: E402

ROWS = [
    ("the box bottom is the robot",              "drives_floor",           "O1"),
    ("...pushed out a fixed 30.9 cm",            "drives_floor",           "O2"),
    ("...predicted from the robot's shape",      "drives_floor",           "F4"),
    ("   (the same thing again, another run)",   "drives_commissioning_v3", "K0"),
    ("+ each camera's own commissioned noise",   "drives_commissioning_v2", "K1"),
    ("+ noise re-measured from the drive",       "drives_commissioning_v3", "K2"),
    ("+ the error the cameras make together",    "drives_commissioning_v4", "K3"),
]


def score(root: str, arm: str):
    found = sorted(glob.glob(
        f"logs/studies/fusion_on_fixed_routes/{root}/fusion_network_traverse/{arm}"
        f"/seed0/experiment_*"))
    if not found:
        return None
    run = Path(found[-1])
    if not (run / "run_summary.json").exists():
        return None
    table = A.rows(run)

    belief = A.aligned_error_cm(run, "belief", table)
    err = belief["aligned_cm"][np.isfinite(belief["aligned_cm"])]
    err_logtime = belief["logtime_cm"][np.isfinite(belief["logtime_cm"])]

    cov = np.array([[[A._float(r, "planner_cov_x"), A._float(r, "planner_cov_xy")],
                     [A._float(r, "planner_cov_xy"), A._float(r, "planner_cov_y")]]
                    for r in table])
    resid = np.stack([belief["gt_x"] - belief["x"], belief["gt_y"] - belief["y"]], axis=1)
    keep = (belief["have"] & np.isfinite(resid).all(axis=1)
            & (cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2 > 0.0))
    nees = A.nees(resid[keep], cov[keep])
    nees = nees[np.isfinite(nees)]
    stated = np.sqrt(np.trace(cov[keep], axis1=1, axis2=2) / 2.0) * 100.0

    state = A.aligned_error_cm(run, "state", table)
    landed = A.landed_mask(state["stamp"])
    corr = state["aligned_cm"][landed & np.isfinite(state["aligned_cm"])]
    counts = A.corrections(run, table)

    return dict(
        cm=float(np.median(corr)) if corr.size else math.nan,
        cp=float(np.percentile(corr, 95)) if corr.size else math.nan,
        bm=float(np.median(err)) if err.size else math.nan,
        bp=float(np.percentile(err, 95)) if err.size else math.nan,
        bm_logtime=float(np.median(err_logtime)) if err_logtime.size else math.nan,
        st=float(np.median(stated)) if stated.size else math.nan,
        nees=float(np.mean(nees)) if nees.size else math.nan,
        nees_med=float(np.median(nees)) if nees.size else math.nan,
        # Kept apart on purpose. `reads` is the count of distinct camera readings and
        # exists only where the drive logs a capture time; `fresh` is log rows with a
        # fresh correction at the 10 Hz log rate, which is an availability fraction
        # times duration. Collapsing them into one column silently compared a count of
        # detections in one row against a count of log rows in the next.
        n=counts["n_detector_rounds"],
        fresh=counts["n_state_publications"],
        rate=counts["state_fresh_rate_hz"],
        blind=counts["longest_gap_s"],
        schema=A.schema_version(run),
    )


def main() -> int:
    out = [(label, arm, score(root, arm)) for label, root, arm in ROWS]
    width = max(len(r[0]) for r in out)
    print(f"{'':{width}}  {'correction':>13} {'belief':>13} {'claims':>8} "
          f"{'NEES':>13} {'reads':>6} {'fresh':>6} {'blind':>6}")
    print(f"{'':{width}}  {'med / p95 cm':>13} {'med / p95 cm':>13} {'med 1sig':>8} "
          f"{'mean / med':>13} {'':>6} {'rows':>6} {'s':>6}")
    for label, _arm, s in out:
        if s is None:
            print(f"{label:{width}}  {'-- drive not yet made --':>13}")
            continue
        print(f"{label:{width}}  {s['cm']:5.2f} /{s['cp']:6.2f} {s['bm']:5.2f} /{s['bp']:6.2f} "
              f"{s['st']:8.2f} {s['nees']:6.2f} /{s['nees_med']:5.2f} "
              f"{(str(s['n']) if s['n'] else '--'):>6} {s['fresh']:6d} "
              f"{s['blind']:6.1f}")
    print(f"\n  NEES targets: 2.0 for the mean, 1.386 for the median. A mean far above "
          f"target\n  beside a median far below it is a heavy tail, not a uniform scale "
          f"error --\n  and a constant added to every measurement is the wrong shape of "
          f"fix for that.")
    print(f"\n  'reads' is distinct camera readings, blank where the drive predates "
          f"capture-time logging.\n  'fresh rows' is log rows holding a fresh correction "
          f"-- an availability fraction, not a count.\n  Scored at log time instead, the "
          f"belief medians would read: "
          + ", ".join(f"{lab.strip()[:18]} {s['bm_logtime']:.2f}"
                      for lab, _a, s in out if s is not None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
