#!/usr/bin/env python3
"""E1 — does the availability field survive a warehouse it was never shown?

Every arm is fitted on `L0` and then frozen.  Each arm is scored against real detector
outcomes in all four environments, and the headline quantity is the *degradation*:
Brier(changed environment) - Brier(`L0`).

For every camera-by-block unit, the arm and calibration link are fitted without that
L0 block, then scored on both the nominal and changed versions of the held-out block.
The GP link is fitted on nested out-of-sample predictions from the remaining blocks.
Thus the paired degradation never compares an in-sample L0 spatial model with an
out-of-environment changed score.  Full-L0 deployment fields are built separately for
the descriptive camera-density and routing analyses.

Run:
    python3 experiments/reconfiguration_holdout/e1_reconfiguration_holdout/run_experiment.py

Writes e1_units.csv, e1_summary.csv, e1_degradation.csv, e1_density.csv and
manifest.json under logs/studies/reconfiguration_holdout/e1_reconfiguration_holdout/.
Reads no ground-truth pose as a model input.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import common as C  # noqa: E402

RESULTS = C.OUT_ROOT / "e1_reconfiguration_holdout"
WORK = C.OUT_ROOT / "work"

#: Bootstrap resamples for the paired-difference interval.  Fixed seed: an interval
#: quoted in a paper must not move between runs of the same script.
BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20260819

#: The arms, in the order the paper's table reports them.  ``per_environment`` marks
#: the arms that are recomputed from the new environment's own inputs; everything else
#: is frozen at its L0 value and is being asked whether that still works.
ARMS = (
    ("constant", "Constant (L0 prevalence)", False, False),
    ("distance", "Distance to camera only", False, False),
    ("fov_range", "FOV / range", False, False),
    ("cad_l0", "CAD raycast, nominal survey", False, True),
    ("cad_env", "CAD raycast, re-surveyed", True, True),
    ("mono_depth", "Monocular depth raycast", True, False),
    ("gp", "GP on L0 detector outcomes", False, False),
    ("hybrid", "Hybrid: GP residual on depth prior", True, False),
)
ARM_LABEL = {k: label for k, label, _, _ in ARMS}
ARM_PER_ENV = {k: per_env for k, _, per_env, _ in ARMS}
ARM_NEEDS_SURVEY = {k: survey for k, _, _, survey in ARMS}


def prepare_l0_events(threshold: float) -> tuple[Path, dict]:
    """Write the L0 training outcomes in the format the canonical GP fitter reads."""
    events_dir = WORK / "events_L0"
    events_dir.mkdir(parents=True, exist_ok=True)
    l0 = C.load_events(C.ENV_BY_KEY["L0"], threshold=threshold)
    for camera, ev in l0.items():
        C.write_events_csv(ev, events_dir / f"{camera}_events.csv")
    xs, ys = C.working_grid()
    np.savez(WORK / "grid.npz", xs=xs, ys=ys)
    return events_dir, l0


def write_prior(env_key: str) -> Path:
    """Per-camera monocular-depth field, in the prior-map form the GP fitter loads."""
    prior_dir = WORK / f"prior_{env_key}"
    prior_dir.mkdir(parents=True, exist_ok=True)
    xs, ys = C.working_grid()
    fields = C.mono_depth_field(env_key)
    for camera, field in fields.items():
        np.savez(prior_dir / f"{camera}_prior.npz", xs=xs, ys=ys,
                 P_mean_map=field, P_conservative_plan_map=field)
    return prior_dir


def fit_gp_field(arm: str, events_dir: Path, out: Path,
                 train_prior_dir: Path | None, query_prior_dir: Path | None) -> dict:
    """Run the GP fitter in its own process and load the field it wrote."""
    cmd = [sys.executable, str(HERE.parent / "gp_fields.py"),
           "--events-dir", str(events_dir), "--arm", arm,
           "--grid-npz", str(WORK / "grid.npz"), "--out", str(out)]
    if arm == "hybrid":
        if train_prior_dir is None or query_prior_dir is None:
            raise RuntimeError("hybrid requires separate L0 training and target query priors")
        cmd += ["--train-prior-dir", str(train_prior_dir),
                "--query-prior-dir", str(query_prior_dir)]
    print(f"[e1] fitting {arm} -> {out.name}")
    # Generated field files used to be keyed only by environment name, which allowed
    # a semantically different residual fit to be silently reused.  Refit these small
    # deployment artifacts on every analysis run; their embedded protocol metadata
    # makes the train/query split inspectable afterwards.
    subprocess.run(cmd, check=True)
    data = np.load(out)
    return {c: np.asarray(data[f"{c}__field"], dtype=float) for c in C.CAMERAS}


def _write_environment_events(env_key: str, threshold: float) -> tuple[Path, dict]:
    events = C.load_events(C.ENV_BY_KEY[env_key], threshold=threshold)
    directory = WORK / f"events_{env_key}"
    directory.mkdir(parents=True, exist_ok=True)
    for camera, ev in events.items():
        C.write_events_csv(ev, directory / f"{camera}_events.csv")
    return directory, events


def _run_transfer_refits(event_dirs: dict[str, Path], prior_dirs: dict[str, Path],
                         environments: list[str]) -> Path:
    """Create fold-clean GP/hybrid predictions for the paired E1 estimand."""
    out = WORK / "gp_transfer_fold_predictions.csv"
    cmd = [
        sys.executable, str(HERE.parent / "gp_transfer_refit.py"),
        "--l0-events-dir", str(event_dirs["L0"]),
        "--block-x-edges", *[str(v) for v in C.BLOCK_X_EDGES],
        "--block-y-edges", *[str(v) for v in C.BLOCK_Y_EDGES],
        "--out", str(out),
    ]
    for key in environments:
        if key != "L0":
            cmd += ["--environment", key, str(event_dirs[key])]
    for key in environments:
        cmd += ["--prior", key, str(prior_dirs[key])]
    print("[e1] fitting fold-clean GP/hybrid transfer predictions")
    subprocess.run(cmd, check=True)
    return out


def build_fields(threshold: float, environments: list[str], *,
                 fit_transfer: bool = False) -> tuple[dict, dict, Path | None]:
    """Build deployment fields and, optionally, fold-clean E1 predictions."""
    events_dir, l0 = prepare_l0_events(threshold)
    event_dirs = {"L0": events_dir}
    for key in environments:
        if key != "L0":
            event_dirs[key], _events = _write_environment_events(key, threshold)
    fields_dir = WORK / "fields"
    fields_dir.mkdir(parents=True, exist_ok=True)

    prior_dirs = {key: write_prior(key) for key in environments}

    shared = {
        "distance": C.distance_field(),
        "fov_range": C.fov_range_field(),
        "cad_l0": C.cad_field(C.ENV_BY_KEY["L0"].world_name),
        "gp": fit_gp_field("gp", events_dir, fields_dir / "gp_L0.npz", None, None),
    }

    per_env: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for key in environments:
        env = C.ENV_BY_KEY[key]
        arms = dict(shared)
        arms["cad_env"] = C.cad_field(env.world_name)
        arms["mono_depth"] = C.mono_depth_field(key)
        arms["hybrid"] = fit_gp_field(
            "hybrid", events_dir, fields_dir / f"hybrid_{key}.npz",
            prior_dirs["L0"], prior_dirs[key])
        per_env[key] = arms

    # The residual is the historical component and must be byte-identical no matter
    # which environment prior it is composed with.  Assert the persisted artifact,
    # not merely the arguments passed above, so a fitter regression fails closed.
    residual_reference: dict[str, np.ndarray] | None = None
    for key in environments:
        artifact = np.load(fields_dir / f"hybrid_{key}.npz")
        residual = {
            camera: np.asarray(artifact[f"{camera}__residual_latent"], dtype=float)
            for camera in C.CAMERAS
        }
        if residual_reference is None:
            residual_reference = residual
        elif any(not np.array_equal(residual[camera], residual_reference[camera])
                 for camera in C.CAMERAS):
            raise RuntimeError(
                f"hybrid residual changed while composing environment {key}; "
                "the residual must be frozen from L0")
    transfer = (_run_transfer_refits(event_dirs, prior_dirs, environments)
                if fit_transfer else None)
    return per_env, l0, transfer


def score_static_arm(arm: str, env_key: str, train_fields: dict, test_fields: dict,
                     l0: dict, events: dict, cameras: tuple[str, ...]) -> list[dict]:
    """Score a non-learned arm with one fold-matched L0 link per paired unit."""
    units = []
    for camera in cameras:
        train_ev = l0[camera]
        test_ev = events[camera]
        train_blocks = C.block_ids(train_ev["xy"])
        test_blocks = C.block_ids(test_ev["xy"])
        for fold in range(C.N_BLOCKS):
            train = train_blocks != fold
            test = test_blocks == fold
            if not test.any() or not train.any():
                continue
            if arm == "constant":
                p_test = np.full(int(test.sum()), float(np.mean(train_ev["hit"][train])))
                a = b = float("nan")
            else:
                raw_train = C.sample_at(train_fields[arm][camera], train_ev["xy"][train])
                a, b = C.fit_link(raw_train, train_ev["hit"][train])
                raw_test = C.sample_at(test_fields[arm][camera], test_ev["xy"][test])
                p_test = C.apply_link(raw_test, a, b)
            m = C.score_predictions(test_ev["hit"][test], p_test)
            units.append({"environment": env_key, "arm": arm, "camera": camera,
                          "fold": fold, "n": int(test.sum()),
                          "link_a": a, "link_b": b, **m})
    return units


def score_transfer_predictions(path: Path, environments: list[str]) -> list[dict]:
    """Fit nested-OOS links and score fold-clean GP/hybrid transfer predictions."""
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    units: list[dict] = []
    for arm in ("gp", "hybrid"):
        p_col = f"p_{arm}"
        for camera in C.CAMERAS:
            for fold in range(C.N_BLOCKS):
                train = [r for r in rows
                         if r["camera"] == camera
                         and int(r["outer_fold"]) == fold
                         and r["role"] == "train_oos"]
                if not train:
                    raise RuntimeError(f"no nested link rows for {arm} {camera} fold {fold}")
                a, b = C.fit_link(
                    np.asarray([float(r[p_col]) for r in train]),
                    np.asarray([float(r["hit"]) for r in train]),
                )
                for env_key in environments:
                    test = [r for r in rows
                            if r["camera"] == camera
                            and int(r["outer_fold"]) == fold
                            and r["role"] == "test"
                            and r["environment"] == env_key]
                    if not test:
                        raise RuntimeError(
                            f"no test rows for {arm} {env_key} {camera} fold {fold}")
                    target = np.asarray([float(r["hit"]) for r in test])
                    raw = np.asarray([float(r[p_col]) for r in test])
                    metrics = C.score_predictions(target, C.apply_link(raw, a, b))
                    units.append({
                        "environment": env_key,
                        "arm": arm,
                        "camera": camera,
                        "fold": fold,
                        "n": len(test),
                        "link_a": a,
                        "link_b": b,
                        **metrics,
                    })
    return units


def bootstrap_ci(values: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap over paired units, dropping units that carry no value.

    A third of the camera-by-block units have no detection at all -- each camera never
    sees the far half of the warehouse -- so their skill score is undefined (the
    climatology term is zero).  Resampling with those in produces a NaN interval, which
    is how this was caught.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, v.size, size=(BOOTSTRAP, v.size))
    means = v[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(diffs: np.ndarray) -> float:
    """Two-sided sign test on paired differences, exact binomial."""
    from math import comb
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    nz = d[d != 0.0]
    n = len(nz)
    if n == 0:
        return 1.0
    k = int(np.sum(nz > 0))
    k = min(k, n - k)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / 2.0 ** n
    return float(min(1.0, 2.0 * tail))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--threshold", type=float, default=C.PRIMARY_THRESHOLD)
    ap.add_argument("--environments", nargs="+", default=None,
                    help="which environments to score; default: every one with a capture")
    args = ap.parse_args(argv)

    RESULTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    available = [e.key for e in C.ENVIRONMENTS
                 if (e.capture / "perception_targets.csv").is_file()]
    wanted = args.environments or available
    missing = [k for k in wanted if k not in available]
    if missing:
        if args.environments is not None:
            raise SystemExit(
                f"explicit environment set is incomplete; missing scored captures for {missing}")
        print(f"[e1] no capture yet for {missing}; scoring {sorted(set(wanted) - set(missing))}")
    wanted = [k for k in wanted if k in available]
    if "L0" not in wanted:
        raise SystemExit("L0 is the development environment and must be present")
    print(f"[e1] environments: {wanted}  threshold {args.threshold}")

    per_env_fields, l0, transfer_path = build_fields(
        args.threshold, wanted, fit_transfer=True)
    if transfer_path is None:
        raise RuntimeError("fold-clean transfer predictions were not generated")

    events = {k: C.load_events(C.ENV_BY_KEY[k], threshold=args.threshold) for k in wanted}
    for k in wanted:
        n = sum(len(events[k][c]["hit"]) for c in C.CAMERAS)
        rate = float(np.mean(np.concatenate([events[k][c]["hit"] for c in C.CAMERAS])))
        vis = float(np.mean(np.concatenate([events[k][c]["oracle"] for c in C.CAMERAS])))
        print(f"[e1] {k:7s} {n:6d} samples, detector hit rate {rate:.4f}, "
              f"oracle-visible fraction {vis:.4f}")

    prevalence = {c: float(np.mean(l0[c]["hit"])) for c in C.CAMERAS}

    units: list[dict] = []
    for arm, _label, _pe, _sv in ARMS:
        if arm in ("gp", "hybrid"):
            continue
        for key in wanted:
            units += score_static_arm(
                arm, key, per_env_fields["L0"], per_env_fields[key],
                l0, events[key], C.CAMERAS)
    units += score_transfer_predictions(transfer_path, wanted)

    unit_cols = ["environment", "arm", "camera", "fold", "n", "link_a", "link_b",
                 "brier", "brier_climatology", "brier_skill",
                 "logloss", "auroc", "ece", "false_visible_rate",
                 "pred_mean", "target_mean"]
    with (RESULTS / "e1_units.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=unit_cols)
        w.writeheader()
        for row in units:
            w.writerow({k: row.get(k, "") for k in unit_cols})

    # Persist the raw-Brier falsifier interactions rather than deriving a headline
    # manually or substituting Brier skill after seeing the changed prevalence.
    for env_key in wanted:
        if env_key == "L0":
            continue
        subprocess.run([
            sys.executable,
            str(HERE / "summarize_inference.py"),
            "--units", str(RESULTS / "e1_units.csv"),
            "--environment", env_key,
            "--out", str(RESULTS / f"e1_inference_{env_key}.csv"),
            "--manifest", str(RESULTS / f"e1_inference_{env_key}_manifest.json"),
        ], check=True)

    # Summary: mean over the 24 camera-fold units of each (arm, environment).
    summary: list[dict] = []
    by_key: dict[tuple[str, str], list[dict]] = {}
    for row in units:
        by_key.setdefault((row["arm"], row["environment"]), []).append(row)
    for (arm, env_key), rows in by_key.items():
        entry = {"arm": arm, "label": ARM_LABEL[arm], "environment": env_key,
                 "recomputed_per_environment": ARM_PER_ENV[arm],
                 "needs_surveyed_model": ARM_NEEDS_SURVEY[arm],
                 "n_units": len(rows)}
        for metric in ("brier", "brier_skill", "logloss", "auroc", "ece",
                       "false_visible_rate"):
            vals = np.array([r[metric] for r in rows], dtype=float)
            entry[f"{metric}_mean"] = float(np.nanmean(vals))
            entry[f"{metric}_std"] = float(np.nanstd(vals))
        summary.append(entry)
    summary.sort(key=lambda e: (e["environment"], e["brier_mean"]))
    with (RESULTS / "e1_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    # Degradation: paired by (camera, fold) against the same arm's L0 unit.
    degradation: list[dict] = []
    for arm, _label, _pe, _sv in ARMS:
        base = {(r["camera"], r["fold"]): r for r in by_key.get((arm, "L0"), [])}
        for env_key in wanted:
            if env_key == "L0":
                continue
            rows = by_key.get((arm, env_key), [])
            paired = [r for r in rows if (r["camera"], r["fold"]) in base]
            diffs = np.array([r["brier"] - base[(r["camera"], r["fold"])]["brier"]
                              for r in paired], dtype=float)
            # Skill loss remains a secondary normalization because its denominator
            # changes with environment prevalence.  The preregistered primary is the
            # raw paired Brier degradation above.
            skill = np.array([base[(r["camera"], r["fold"])]["brier_skill"] - r["brier_skill"]
                              for r in paired], dtype=float)
            lo, hi = bootstrap_ci(diffs)
            slo, shi = bootstrap_ci(skill)
            degradation.append({
                "arm": arm, "label": ARM_LABEL[arm], "environment": env_key,
                "n_units": len(diffs),
                "brier_L0": float(np.nanmean([r["brier"] for r in by_key[(arm, "L0")]])),
                "brier_env": float(np.nanmean([r["brier"] for r in rows])),
                "delta_brier": float(np.nanmean(diffs)),
                "ci95_low": lo, "ci95_high": hi,
                "skill_L0": float(np.nanmean([r["brier_skill"] for r in by_key[(arm, "L0")]])),
                "skill_env": float(np.nanmean([r["brier_skill"] for r in rows])),
                "skill_lost": float(np.nanmean(skill)),
                "skill_ci95_low": slo, "skill_ci95_high": shi,
                "n_units_skill": int(np.sum(np.isfinite(skill))),
                "units_worse": int(np.sum(skill[np.isfinite(skill)] > 0)),
                "units_worse_brier": int(np.sum(diffs[np.isfinite(diffs)] > 0)),
                "sign_test_p_two_sided": sign_test_p(skill),
                "sign_test_p_brier": sign_test_p(diffs),
            })
    with (RESULTS / "e1_degradation.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(degradation[0].keys()))
        w.writeheader()
        w.writerows(degradation)

    # Camera density: fused availability over each subset, scored on whether ANY
    # camera in the subset detected the robot at that pose.
    density: list[dict] = []
    for subset_name, cams in C.CAMERA_SUBSETS.items():
        for arm, _label, _pe, _sv in ARMS:
            if arm == "constant":
                continue
            for env_key in wanted:
                fields = per_env_fields[env_key]
                ev = events[env_key]
                train = l0
                # Pair by the actual pose, not by array position.  A capture can be
                # short a few frames -- the L1 capture is missing 39 of 15,072 -- and
                # then the per-camera arrays have different lengths and different
                # contents, so index-aligned fusion would silently combine one
                # camera's pose with another's.  Fuse over the poses every camera in
                # the subset actually delivered.
                per_cam: dict[str, dict] = {}
                for c in cams:
                    raw_tr = C.sample_at(per_env_fields["L0"][arm][c], train[c]["xy"])
                    a, b = C.fit_link(raw_tr, train[c]["hit"])
                    p = C.apply_link(C.sample_at(fields[arm][c], ev[c]["xy"]), a, b)
                    per_cam[c] = {
                        (round(float(x), 4), round(float(y), 4), round(float(t), 4)):
                            (float(h), float(pp))
                        for (x, y), t, h, pp in zip(ev[c]["xy"], ev[c]["theta"],
                                                    ev[c]["hit"], p)
                    }
                shared = set(per_cam[cams[0]])
                for c in cams[1:]:
                    shared &= set(per_cam[c])
                if not shared:
                    continue
                poses = sorted(shared)
                hits = np.array([max(per_cam[c][k][0] for c in cams) for k in poses])
                p_none = np.ones(len(poses), dtype=float)
                for c in cams:
                    p_none *= np.array([1.0 - per_cam[c][k][1] for k in poses])
                p_any = np.clip(1.0 - p_none, 1e-6, 1 - 1e-6)
                m = C.score_predictions(hits, p_any)
                density.append({"subset": subset_name, "n_cameras": len(cams),
                                "arm": arm, "label": ARM_LABEL[arm],
                                "environment": env_key, "n": len(poses), **m})
    if density:
        with (RESULTS / "e1_density.csv").open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(density[0].keys()))
            w.writeheader()
            w.writerows(density)

    (RESULTS / "manifest.json").write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "headings": list(C.THETAS),
        "environments": {k: {"world": C.ENV_BY_KEY[k].world_name,
                             "layout": C.ENV_BY_KEY[k].layout,
                             "lighting": C.ENV_BY_KEY[k].lighting,
                             "capture": str(C.ENV_BY_KEY[k].capture.relative_to(C.REPO)),
                             "n_samples": int(sum(len(events[k][c]["hit"]) for c in C.CAMERAS))}
                         for k in wanted},
        "arms": [{"key": k, "label": lb, "recomputed_per_environment": pe,
                  "needs_surveyed_model": sv} for k, lb, pe, sv in ARMS],
        "l0_prevalence": prevalence,
        "bootstrap": {"resamples": BOOTSTRAP, "seed": BOOTSTRAP_SEED},
        "blocks": {"x_edges": list(C.BLOCK_X_EDGES), "y_edges": list(C.BLOCK_Y_EDGES)},
        "paired_protocol": {
            "name": "fold_matched_nominal_to_changed_v2",
            "model_fit": "L0 excluding the paired outer spatial block",
            "link_fit_static": "L0 excluding the paired outer spatial block",
            "link_fit_gp_hybrid": (
                "nested out-of-sample predictions from L0 blocks excluding the "
                "outer block and each predicted inner block"),
            "hybrid_residual": (
                "fit against the L0 monocular prior once; add unchanged residual "
                "to each environment's query prior"),
            "audit_note": (
                "v1 only held the calibration link out on L0 and recomputed the "
                "hybrid residual against each target prior; v2 corrects both issues"),
        },
        "hybrid_residual_sha256": {
            camera: hashlib.sha256(
                np.asarray(
                    np.load(WORK / "fields/hybrid_L0.npz")[
                        f"{camera}__residual_latent"], dtype=float
                ).tobytes()
            ).hexdigest()
            for camera in C.CAMERAS
        },
    }, indent=2), encoding="utf-8")

    print("\n[e1] held-out Brier (lower better) and skill against each environment's "
          "own base rate (higher better)")
    print("  " + f"{'arm':36s}" + "".join(f"{k + ' Brier':>13s}{k + ' skill':>13s}"
                                          for k in wanted))
    for arm, label, _pe, _sv in ARMS:
        row = f"  {label:36s}"
        for k in wanted:
            m = [e for e in summary if e["arm"] == arm and e["environment"] == k]
            if m:
                row += f"{m[0]['brier_mean']:13.4f}{m[0]['brier_skill_mean']:13.4f}"
            else:
                row += f"{'-':>13s}{'-':>13s}"
        print(row)
    print("\n[e1] skill LOST between the nominal and each changed environment "
          "(positive = the estimator got worse)")
    for row in sorted(degradation, key=lambda r: -r["skill_lost"]):
        print(f"  {row['label']:36s} {row['environment']:7s} "
              f"{row['skill_lost']:+.4f}  95% [{row['skill_ci95_low']:+.4f}, "
              f"{row['skill_ci95_high']:+.4f}]  worse on {row['units_worse']}/"
              f"{row['n_units_skill']} informative units  p={row['sign_test_p_two_sided']:.4f}")
    print(f"\n[e1] wrote {RESULTS.relative_to(C.REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
