"""Re-fit the measurement covariance on schema-5 drives at the fusion speed.

The two-term model `R = scale^2 * C_stated + sigma_m^2 * I` was fitted once already, in
`per_camera_error.py`, on the pre-repair commissioning drives at 1 m/s. It returned a metric
floor of 1.25 cm -- which independently matches the floor measured by range asymptote in
`docs/ANSWERS.md` Q6 -- and was then REJECTED by its own pre-registered rule, because a floor
shared by every camera pushes A and E from conservative to far too conservative.

So the question here is NOT "re-fit the same uniform floor on better data". It is whether the
floor has to be **per camera**. That is arm (d).

DECISION RULE, fixed before the campaign that feeds this existed
(also recorded in docs/ANSWERS.md, "the rule written before the data arrives"):

  arms      (a) as deployed          scale = 1, floor = 0
            (b) pixel-only refit     fit scale, floor = 0
            (c) uniform floor        fit scale and one shared floor
            (d) per-camera floor     fit one shared scale and one floor per camera

  held out  the whole of `fusion_network_traverse`, named before any fit was run. Fitting
            uses the other three routes only.

  pass      every camera's NSE inside [0.5, 2.0] on the held-out route -- where NSE = 2.0 is
            the CONSISTENT value, above is overconfident, below is conservative
            AND readings beyond 4 sigma (Mahalanobis) below 2%. NSE is a median statistic
            and is blind to the tail; the tail is the reason this question exists, so an arm
            that fixes the bulk and leaves the tail does not pass.

  null      if (d) does not beat (c) on the held-out route, the floor is uniform after all
            and the earlier rejection was a small-sample artifact. That is a result. Say so.

  forbidden choosing the floor by scanning it against the NEES it will be judged by. The
            floor is a physical quantity -- the error the hull model makes that pixel noise
            does not describe -- and Q6 measures it as the asymptote of error versus range.

Only the hull arms (F1-F4) are used. O1 and O2 misread the box on purpose, so their residuals
describe those broken observation models rather than the estimator.

    python3 experiments/learned_measurement_covariance/refit_schema5.py [campaign_dir]
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "experiments/learned_measurement_covariance"))
sys.path.insert(0, str(REPO / "experiments/fusion_on_fixed_routes"))
import aligned as A  # noqa: E402
from per_camera_error import (  # noqa: E402
    CAMS, CHI2_2_MEDIAN, SIGMA_PX, _nees, _nll, _stack, fit_two_term,
)

CAMPAIGN = REPO / "logs/studies/fusion_on_fixed_routes/diagnostic_schema5_20260831"
HELD_OUT_ROUTE = "fusion_network_traverse"      # named before any fit was run
HULL_ARMS = ("F1", "F2", "F3", "F4")
OUT = REPO / "logs/studies/learned_measurement_covariance/refit_schema5.json"
FLOORS = np.arange(0.0, 0.0601, 0.0005)
SCALES = np.arange(0.5, 3.01, 0.05)


def load(campaign: Path) -> list[dict]:
    """Every admitted reading from every hull-arm drive, scored at its own capture time."""
    rows = []
    for manifest in sorted(campaign.rglob("run_manifest.json")):
        run = manifest.parent
        parts = run.relative_to(campaign).parts
        if len(parts) < 2 or parts[1] not in HULL_ARMS:
            continue
        route, arm = parts[0], parts[1]
        schema = A.schema_version(run)
        for e in A.readings(run, admitted_only=True, dedupe=True):
            cov = e["cov"]
            if not np.isfinite(cov).all() or np.linalg.det(cov) <= 0.0:
                continue
            cam = CAMS.get(f"camera_{e['camera']}")
            if cam is None:
                continue
            rows.append(dict(camera=e["camera"], route=route, arm=arm, schema=schema,
                             cov=cov, error=e["error"], range_m=e["range_m"],
                             pred_h_px=e.get("pred_h_px", math.nan),
                             bbox_h_px=e.get("bbox_h_px", math.nan)))
    return rows


def nse_by_camera(rows, scale=1.0, floor_m=0.0, floors_by_cam=None) -> dict:
    out = {}
    for cam in sorted({r["camera"] for r in rows}):
        sub = [r for r in rows if r["camera"] == cam]
        f = floors_by_cam[cam] if floors_by_cam else floor_m
        out[cam] = float(np.median(_nees(sub, scale, f)) / CHI2_2_MEDIAN * 2.0)
    return out


def tail_rate(rows, scale=1.0, floor_m=0.0, floors_by_cam=None) -> float:
    """Fraction of readings beyond 4 sigma in Mahalanobis distance (NEES > 16)."""
    if floors_by_cam:
        q = np.concatenate([_nees([r for r in rows if r["camera"] == c], scale, floors_by_cam[c])
                            for c in sorted({r["camera"] for r in rows})])
    else:
        q = _nees(rows, scale, floor_m)
    return float(np.mean(q > 16.0))


def fit_per_camera_floor(rows):
    """One shared pixel scale, one metric floor per camera. Joint over the shared scale."""
    cams = sorted({r["camera"] for r in rows})
    by_cam = {c: _stack([r for r in rows if r["camera"] == c]) for c in cams}
    best = (math.inf, 1.0, None)
    for scale in SCALES:
        total, floors = 0.0, {}
        for c in cams:
            n = len(by_cam[c][1])
            vals = [(_nll(by_cam[c], float(scale), float(f)), float(f)) for f in FLOORS]
            v, f = min(vals)
            total += v * n
            floors[c] = f
        if total < best[0]:
            best = (total, float(scale), floors)
    return dict(scale=best[1], sigma_px=best[1] * SIGMA_PX,
                floors_cm={c: v * 100 for c, v in best[2].items()}, floors_m=best[2])


def main() -> int:
    campaign = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else CAMPAIGN)
    rows = load(campaign)
    if not rows:
        raise SystemExit(f"no hull-arm readings under {campaign}")
    schemas = sorted({r["schema"] for r in rows})
    fit = [r for r in rows if r["route"] != HELD_OUT_ROUTE]
    held = [r for r in rows if r["route"] == HELD_OUT_ROUTE]
    print(f"{len(rows)} admitted hull readings, schema {schemas}")
    print(f"  fit   {len(fit):6d} from {sorted({r['route'] for r in fit})}")
    print(f"  held  {len(held):6d} from {HELD_OUT_ROUTE}")
    if not held or not fit:
        raise SystemExit("need both a fit set and the held-out route")

    px = fit_two_term(fit, floors=np.array([0.0]), scales=SCALES)
    uni = fit_two_term(fit, floors=FLOORS, scales=SCALES)
    per = fit_per_camera_floor(fit)
    print(f"\nfitted on {len(fit)} readings:")
    print(f"  pixel-only     sigma_px {px['sigma_px']:.3f}")
    print(f"  uniform floor  sigma_px {uni['sigma_px']:.3f}  floor {uni['floor_cm']:.2f} cm")
    print(f"  per-camera     sigma_px {per['sigma_px']:.3f}  floors "
          + ", ".join(f"{c} {v:.2f}" for c, v in sorted(per['floors_cm'].items())) + " cm")

    arms = {
        "as_deployed":   dict(scale=1.0, floor_m=0.0, floors=None),
        "pixel_refit":   dict(scale=px["scale"], floor_m=0.0, floors=None),
        "uniform_floor": dict(scale=uni["scale"], floor_m=uni["floor_m"], floors=None),
        "per_camera_floor": dict(scale=per["scale"], floor_m=0.0, floors=per["floors_m"]),
    }
    print(f"\nHELD OUT ({HELD_OUT_ROUTE}) -- NSE 2.0 is consistent, >2.0 overconfident")
    results = {}
    for name, cfg in arms.items():
        nse = nse_by_camera(held, cfg["scale"], cfg["floor_m"], cfg["floors"])
        tail = tail_rate(held, cfg["scale"], cfg["floor_m"], cfg["floors"])
        ok_nse = all(0.5 <= v <= 2.0 for v in nse.values())
        ok_tail = tail < 0.02
        results[name] = dict(nse=nse, tail_beyond_4sigma=tail,
                             passes_nse=ok_nse, passes_tail=ok_tail,
                             passes=bool(ok_nse and ok_tail), **{
                                 k: v for k, v in cfg.items() if k != "floors"})
        row = "  ".join(f"{c} {v:5.2f}" for c, v in sorted(nse.items()))
        print(f"  {name:18s} {row}   tail {100*tail:5.2f}%   "
              f"{'PASS' if ok_nse and ok_tail else 'fail'}"
              f"{'' if ok_nse else ' (nse)'}{'' if ok_tail else ' (tail)'}")

    winner = next((n for n in ("per_camera_floor", "uniform_floor", "pixel_refit",
                               "as_deployed") if results[n]["passes"]), None)
    verdict = ("no arm passes" if winner is None else f"{winner} passes")
    if results["per_camera_floor"]["passes"] and not results["uniform_floor"]["passes"]:
        verdict += "; the floor must be per camera"
    elif results["uniform_floor"]["passes"]:
        verdict += "; a uniform floor is enough -- the earlier rejection was small-sample"
    print(f"\nVERDICT: {verdict}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        campaign=str(campaign.relative_to(REPO)) if campaign.is_relative_to(REPO)
                 else str(campaign), schemas=schemas,
        held_out_route=HELD_OUT_ROUTE, n_fit=len(fit), n_held_out=len(held),
        sigma_px_deployed=SIGMA_PX, pixel_refit=px, uniform=uni, per_camera=per,
        arms=results, verdict=verdict,
        rule="every camera NSE in [0.5,2.0] on the held-out route AND beyond-4-sigma < 2%",
    ), indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
