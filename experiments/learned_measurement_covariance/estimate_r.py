"""Let the camera network measure its own noise, with no ground truth anywhere.

The idea: when two cameras see the robot at the same instant, the difference between
their two answers is a quantity we can watch without knowing where the robot actually
is.  Its size is set by the two cameras' own noise and nothing else:

    spread of (answer_c - answer_d)  =  noise of c  +  noise of d

With five cameras overlapping there are ten such pairs, which is far more equations
than unknowns -- so each camera's noise can be solved for individually.  This is the
same trick a set of clocks uses to work out which of them is drifting.

What it CANNOT see: a lean shared by every camera.  Differences cancel it.  That is a
real limit and it is stated in the results, not hidden.

Ground truth is read at the very end for SCORING ONLY, to check the answer.  It is
never an input to the estimate.
"""
import csv, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fusion_on_fixed_routes"))
import aligned as A  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SIGMA_PX = 0.7642679454946852          # the one number commissioning hands the runtime
CHI2_MEDIAN_2D = 1.3862943611198906    # median of chi-square with 2 degrees of freedom


def load(drives):
    """Every logged camera answer, grouped by the DETECTOR ROUND that produced it.

    Grouped by capture time, not by the manager's decision stamp. The manager decides at
    20 Hz against a 5 Hz detector, so grouping by decision splits one round into about
    four identical groups: measured 3.99 rows per (camera, capture time). The estimator
    below is a spread over groups, so replicated groups leave the estimate roughly where
    it was but quote it with about twice the confidence the data supports, and weight a
    round by however many times the manager happened to republish it.

    Drives that predate capture-time logging fall back to the decision stamp, and say so
    through the returned flag, because on those the split cannot be undone.
    """
    groups = {}
    exact = True
    for drive in drives:
        path = Path(drive) / "fusion_observations.csv"
        if not path.exists():
            continue
        # The truth path, so a reading can be scored at the instant the CAMERA saw it
        # rather than at logging time. `gt_x`/`gt_y` in this file is the truth held when
        # the decision reached the logger, later by the whole pipeline delay -- a delay
        # identical on every camera, so it reads as measurement error on all of them.
        try:
            truth_series = A.truth_series(Path(drive))
        except SystemExit:
            truth_series = None
        for row in csv.DictReader(open(path)):
            try:
                cov = np.array([[float(row["obs_cov_xx"]), float(row["obs_cov_xy"])],
                                [float(row["obs_cov_xy"]), float(row["obs_cov_yy"])]])
                xy = np.array([float(row["obs_x"]), float(row["obs_y"])])
                gt = np.array([float(row["gt_x"]), float(row["gt_y"])])
            except (KeyError, ValueError):
                continue
            if not np.isfinite(cov).all() or np.linalg.det(cov) <= 0.0:
                continue
            try:
                capture = float(row["obs_stamp"])
            except (KeyError, TypeError, ValueError):
                capture = float("nan")
            if math.isfinite(capture):
                key = (str(drive), "cap", round(capture, 6))
                if truth_series is not None:
                    tx, ty = truth_series.at([capture])
                    if np.isfinite(tx[0]) and np.isfinite(ty[0]):
                        gt = np.array([float(tx[0]), float(ty[0])])
            else:
                exact = False
                key = (str(drive), "dec", round(float(row["stamp"]), 4))
            # first write wins: the reading as the manager first computed it
            groups.setdefault(key, {}).setdefault(row["camera"], (xy, cov, gt))
    if not exact:
        print("  note: at least one drive logs no capture time, so its rounds are grouped "
              "by the manager's decision stamp and each is counted about four times.")
    return groups


def estimate(groups, cameras):
    """Solve for each camera's noise from how far it sits from what the others saw.

    At an instant where N cameras all report, take each camera's distance from the average
    of all N. Whatever error they share cancels in that subtraction -- the truth does too,
    which is why this needs no ground truth. What is left is the camera's own noise, mixed
    with a known fraction of everyone else's:

        spread of (camera c - the average)  =  (1-1/N)^2 * noise of c  +  (1/N^2) * sum of the rest

    One equation per camera per instant, solved together for all five.
    """
    index = {c: i for i, c in enumerate(cameras)}

    # A camera can sit consistently to one side of the others. That is a lean, not noise,
    # and inflating the noise to cover it would be the wrong fix -- so it is measured and
    # removed first, and reported separately.
    lean_sum, lean_n = {}, {}
    for members in groups.values():
        if len(members) < 2:
            continue
        mean = np.mean([v[0] for v in members.values()], axis=0)
        for c, (xy, _cov, _gt) in members.items():
            lean_sum[c] = lean_sum.get(c, np.zeros(2)) + (xy - mean)
            lean_n[c] = lean_n.get(c, 0) + 1
    lean = {c: lean_sum[c] / lean_n[c] for c in lean_sum}

    # Match MEDIANS, not sums of squares. A least-squares fit to squared deviations is pulled
    # by the few large ones -- on the first drives it overstated one camera by 3x and understated
    # another by half, and applying those numbers made a well-stated camera overconfident. The
    # median is unmoved by the tail, so each camera's scale is nudged until its typical deviation
    # is the size the model says it should be, and the nudges are repeated until they stop.
    samples = []
    for members in groups.values():
        n = len(members)
        if n < 2:
            continue
        mean = np.mean([v[0] for v in members.values()], axis=0)
        covs = {c: v[1] for c, v in members.items()}
        for c, (xy, _cov, _gt) in members.items():
            d = (xy - mean) - lean[c]
            samples.append((c, d, covs, n))

    # The ellipses are far from round -- a camera is much surer across its viewing ray than
    # along it -- so the deviation has to be measured against the ellipse's actual shape, not
    # against its average width. Using the full shape is what makes the answer agree with what
    # ground truth says afterwards; using the average width put one camera out by six times.
    k = np.ones(len(cameras))
    for _ in range(80):
        per = {c: [] for c in cameras}
        for c, d, covs, n in samples:
            S = (1.0 - 1.0 / n) ** 2 * k[index[c]] * covs[c]
            for o, cov_o in covs.items():
                if o != c:
                    S = S + (1.0 / n) ** 2 * k[index[o]] * cov_o
            if np.linalg.det(S) > 0.0:
                per[c].append(float(d @ np.linalg.solve(S, d)))
        step = np.ones(len(cameras))
        for c in cameras:
            if len(per[c]) >= 20:
                step[index[c]] = float(np.median(per[c])) / CHI2_MEDIAN_2D
        if np.all(np.abs(step - 1.0) < 1.0e-4):
            break
        k = np.clip(k * step ** 0.5, 1.0e-4, 1.0e4)
    return k, lean, len(samples)


def _pairs(members, cameras):
    present = [c for c in cameras if c in members]
    return [(present[i], present[j])
            for i in range(len(present)) for j in range(i + 1, len(present))]


def score_against_truth(groups, cameras, k):
    """SCORING ONLY. What each camera's noise really was, and what the estimate said."""
    out = {}
    for c in cameras:
        raw, fixed, err = [], [], []
        for members in groups.values():
            if c not in members:
                continue
            xy, cov, gt = members[c]
            d = gt - xy
            raw.append(float(d @ np.linalg.solve(cov, d)))
            fixed.append(float(d @ np.linalg.solve(cov * max(k[cameras.index(c)], 1e-6), d)))
            err.append(float(np.linalg.norm(d)) * 100.0)
        if raw:
            out[c] = dict(n=len(raw), before=float(np.median(raw)),
                          after=float(np.median(fixed)), err_cm=float(np.median(err)))
    return out


def write_artifact(cameras, k, lean, n_eq, out_path, pooled=SIGMA_PX):
    """Emit a calibration file the runtime can read directly.

    Same shape as the commissioning artifact, so pointing the manager at this file is the
    whole of "recommission the network" -- no new parameter, no new code path. What differs
    is where the numbers came from: camera disagreement on a drive, not a survey against
    ground truth in a lab.
    """
    import json
    by_camera = {}
    for c, ki in zip(cameras, k):
        name = c if c.startswith("camera_") else f"camera_{c}"
        by_camera[name] = pooled * math.sqrt(max(float(ki), 1e-6))
    payload = {
        "created_by": "experiments/learned_measurement_covariance/estimate_r.py",
        "how": ("each camera's distance from what the others saw at the same instant; "
                "no ground truth of any kind"),
        "cannot_see": ("a lean shared by every camera -- differences cancel it. That part "
                       "still needs a survey, or a robot at a known place."),
        "equations": int(n_eq),
        "calibration": {
            "sigma_px": float(pooled),
            "sigma_px_by_camera": {k2: float(v) for k2, v in sorted(by_camera.items())},
            "lean_from_network_mean_cm": {
                (c if c.startswith("camera_") else f"camera_{c}"):
                    float(np.linalg.norm(lean[c]) * 100.0)
                for c in sorted(lean)},
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--write=")]
    write_to = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--write=")), None)
    drives = args
    if not drives:
        raise SystemExit("usage: estimate_r.py [--write=out.json] <drive dir> ...")
    groups = load(drives)
    cameras = sorted({c for m in groups.values() for c in m})
    k, offset, n_eq = estimate(groups, cameras)

    print(f"{len(groups)} instants, {len(cameras)} cameras, {n_eq} pairwise comparisons\n")
    print("What the network says about itself (no ground truth used):")
    print(f"  {'camera':8} {'noise is':>10} {'stated pixel':>13} {'should be':>11}")
    for c, ki in zip(cameras, k):
        print(f"  {c:8} {ki:9.1f}x {SIGMA_PX:12.2f} px {SIGMA_PX*math.sqrt(max(ki,0)):10.2f} px")

    print("\n  each camera's lean away from what the others saw (cm) -- visible without truth;")
    print("  a lean they ALL share is not, and never can be, from overlap alone:")
    for c, d in sorted(offset.items()):
        print(f"    {c}: {np.linalg.norm(d)*100:5.2f}")

    truth = score_against_truth(groups, cameras, k)
    print("\nScoring against ground truth (evaluation only). Each reading is compared to the\n"
          "truth at the instant the CAMERA saw it, where the drive logs a capture time.\n"
          "Only on a drive that does not is it compared at logging time, which is later by\n"
          "the pipeline delay and charges that shared delay to every camera alike.")
    print(f"  {'camera':8} {'n':>6} {'real error':>11} {'was':>8} {'now':>8}   (1.39 = right)")
    for c in cameras:
        t = truth.get(c)
        if not t:
            continue
        print(f"  {c:8} {t['n']:6d} {t['err_cm']:10.2f} cm {t['before']:8.2f} {t['after']:8.2f}")
    allb = np.median([t["before"] for t in truth.values()])
    alla = np.median([t["after"] for t in truth.values()])
    print(f"\n  typical camera: overconfident by {allb/CHI2_MEDIAN_2D:.1f}x before, "
          f"{alla/CHI2_MEDIAN_2D:.1f}x after")

    if write_to:
        payload = write_artifact(cameras, k, offset, n_eq, write_to)
        listing = ", ".join(f"{a.replace('camera_','')}={b:.3f}"
                            for a, b in payload["calibration"]["sigma_px_by_camera"].items())
        print(f"\nwrote {write_to}\n  recommissioned pixel noise: {listing}")
        print("  point manager_commissioned_calibration_path at this file to use it.")


if __name__ == "__main__":
    main()
