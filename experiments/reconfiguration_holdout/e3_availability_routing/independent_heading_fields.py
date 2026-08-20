#!/usr/bin/env python3
"""Build E3 deployment fields from headings withheld from route evaluation.

The nominal L0 capture has eight robot headings, whereas the reconfigured L1
capture has the four cardinal headings only.  E3 previously built the learned
field and its calibration link from L0's cardinal outcomes and then called those
same outcomes nominal route truth.  That made the nominal half of the
difference-in-differences contrast in-sample.

This module owns E3's independent-heading protocol:

* fit the GP and hybrid residual on L0 diagonal headings only;
* fit every arm's probability link on those same diagonal headings;
* reserve L0 cardinal headings for nominal route evaluation;
* evaluate L1 on the matching cardinal headings; and
* keep all generated events, priors, and fields in an E3-specific cache.

The hybrid calls the study's corrected :mod:`gp_fields` process with separate
L0 training and environment-specific query priors.  Thus the residual is fitted
once against the L0 prior and remains frozen when it is added to the L1 prior.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE = Path(__file__).resolve().parent


def _load_study_common():
    name = "_reconfiguration_holdout_common"
    path = (HERE.parent / "common.py").resolve()
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != path:
            raise ImportError(f"{name} resolves to an unexpected module")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reconfiguration common module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


C = _load_study_common()


PROTOCOL_ID = "e3_independent_heading_v1"

# L1 was captured at these headings.  They are the only headings permitted to
# contribute route truth in either environment.
EVALUATION_HEADINGS = tuple(float(value) for value in C.THETAS)

# The other four headings in L0's eight-heading capture.  They are used for model
# fitting and probability-link calibration, never for route-truth evaluation.
TRAINING_HEADINGS = tuple(
    float((value + math.pi / 4.0) % (2.0 * math.pi)) for value in EVALUATION_HEADINGS
)

WORK = C.OUT_ROOT / "work" / PROTOCOL_ID
EVENTS_DIR = WORK / "events_L0_diagonal"
PRIORS_DIR = WORK / "priors"
FIELDS_DIR = WORK / "fields"
GRID_PATH = WORK / "grid.npz"
FIELD_MANIFEST_PATH = WORK / "field_manifest.json"
GP_FIELDS_SCRIPT = HERE.parent / "gp_fields.py"
LINK_OOF_SCRIPT = HERE / "diagonal_link_oof.py"
LINK_OOF_PATH = WORK / "link_oof" / "gp_hybrid_L0_diagonal_oof.csv"


def _angular_distance(a: float, b: float) -> float:
    """Smallest absolute circular distance between two headings."""

    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


def validate_heading_partition() -> None:
    """Fail closed unless train and evaluation headings are complete and disjoint."""

    if len(TRAINING_HEADINGS) != 4 or len(EVALUATION_HEADINGS) != 4:
        raise RuntimeError("E3 requires four training and four evaluation headings")
    for values, name in (
        (TRAINING_HEADINGS, "training"),
        (EVALUATION_HEADINGS, "evaluation"),
    ):
        if any(
            _angular_distance(values[i], values[j]) < C.THETA_TOL
            for i in range(len(values))
            for j in range(i + 1, len(values))
        ):
            raise RuntimeError(f"E3 {name} headings contain duplicates")
    if any(
        _angular_distance(train, test) < C.THETA_TOL
        for train in TRAINING_HEADINGS
        for test in EVALUATION_HEADINGS
    ):
        raise RuntimeError("E3 training and evaluation headings overlap")


def _assert_event_headings(events: dict[str, dict[str, np.ndarray]],
                           allowed: tuple[float, ...], role: str) -> None:
    """Verify that a loader returned only, and all, headings declared for its role."""

    for camera in C.CAMERAS:
        observed = np.asarray(events[camera]["theta"], dtype=float)
        unexpected = [
            float(theta) for theta in np.unique(observed)
            if not any(_angular_distance(theta, value) < C.THETA_TOL for value in allowed)
        ]
        missing = [
            float(value) for value in allowed
            if not np.any([_angular_distance(theta, value) < C.THETA_TOL
                           for theta in observed])
        ]
        if unexpected or missing:
            raise RuntimeError(
                f"{role} {camera}: heading contract failed; "
                f"unexpected={unexpected}, missing={missing}"
            )


def load_training_events(threshold: float) -> dict[str, dict[str, np.ndarray]]:
    """Load L0 diagonal outcomes, the only detector labels allowed for fitting."""

    validate_heading_partition()
    events = C.load_events(
        C.ENV_BY_KEY["L0"], threshold=threshold, thetas=TRAINING_HEADINGS
    )
    _assert_event_headings(events, TRAINING_HEADINGS, "training")
    return events


def load_evaluation_events(env_key: str, threshold: float) -> dict[str, dict[str, np.ndarray]]:
    """Load cardinal outcomes, the only detector labels allowed as route truth."""

    validate_heading_partition()
    events = C.load_events(
        C.ENV_BY_KEY[env_key], threshold=threshold, thetas=EVALUATION_HEADINGS
    )
    _assert_event_headings(events, EVALUATION_HEADINGS, f"evaluation {env_key}")
    return events


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(C.REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_training_events(threshold: float) -> dict[str, dict[str, np.ndarray]]:
    events = load_training_events(threshold)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    for camera in C.CAMERAS:
        C.write_events_csv(events[camera], EVENTS_DIR / f"{camera}_events.csv")
    WORK.mkdir(parents=True, exist_ok=True)
    xs, ys = C.working_grid()
    np.savez(GRID_PATH, xs=xs, ys=ys)
    return events


def _write_prior(env_key: str) -> Path:
    """Write one environment's image-derived prior into the GP fitter format."""

    directory = PRIORS_DIR / env_key
    directory.mkdir(parents=True, exist_ok=True)
    xs, ys = C.working_grid()
    for camera, field in C.mono_depth_field(env_key).items():
        np.savez(
            directory / f"{camera}_prior.npz",
            xs=xs,
            ys=ys,
            P_mean_map=field,
            P_conservative_plan_map=field,
        )
    return directory


def _run_link_oof(prior_dirs: dict[str, Path]) \
        -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """Fit six spatial refits and load pooled OOF scores for learned-arm links."""

    command = [
        sys.executable,
        str(LINK_OOF_SCRIPT),
        "--events-dir",
        str(EVENTS_DIR),
        "--l0-prior-dir",
        str(prior_dirs["L0"]),
        "--block-x-edges",
        *[str(value) for value in C.BLOCK_X_EDGES],
        "--block-y-edges",
        *[str(value) for value in C.BLOCK_Y_EDGES],
        "--out",
        str(LINK_OOF_PATH),
    ]
    print("[e3 fields] fitting spatially out-of-fold diagonal link scores", flush=True)
    subprocess.run(command, check=True)

    by_arm: dict[str, dict[str, dict[str, list[float]]]] = {
        arm: {
            camera: {"score": [], "hit": [], "block": []}
            for camera in C.CAMERAS
        }
        for arm in ("gp", "hybrid")
    }
    with LINK_OOF_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            camera = str(row["camera"])
            if camera not in C.CAMERAS:
                raise RuntimeError(f"{LINK_OOF_PATH}: unknown camera {camera!r}")
            for arm in ("gp", "hybrid"):
                by_arm[arm][camera]["score"].append(float(row[f"p_{arm}"]))
                by_arm[arm][camera]["hit"].append(float(row["hit"]))
                by_arm[arm][camera]["block"].append(float(row["block"]))

    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for arm in ("gp", "hybrid"):
        result[arm] = {}
        for camera in C.CAMERAS:
            values = by_arm[arm][camera]
            result[arm][camera] = {
                key: np.asarray(column, dtype=float) for key, column in values.items()
            }
            lengths = {len(column) for column in result[arm][camera].values()}
            if len(lengths) != 1 or next(iter(lengths), 0) == 0:
                raise RuntimeError(f"{LINK_OOF_PATH}: incomplete {arm} {camera} rows")
            observed_blocks = set(result[arm][camera]["block"].astype(int).tolist())
            if observed_blocks != set(range(C.N_BLOCKS)):
                raise RuntimeError(
                    f"{LINK_OOF_PATH}: {arm} {camera} blocks {observed_blocks}, "
                    f"expected {set(range(C.N_BLOCKS))}"
                )
    return result


def field_cache_path(arm: str, query_env: str) -> Path:
    """Return an E3-specific cache name that exposes its fit/query provenance."""

    if arm == "gp":
        return FIELDS_DIR / "gp_train_L0_diagonal.npz"
    if arm == "hybrid":
        return FIELDS_DIR / f"hybrid_train_L0_diagonal_query_{query_env}.npz"
    raise ValueError(f"no learned E3 field for arm {arm!r}")


def gp_field_command(arm: str, query_env: str, prior_dirs: dict[str, Path]) -> list[str]:
    """Construct the auditable fitter command used for one learned field."""

    command = [
        sys.executable,
        str(GP_FIELDS_SCRIPT),
        "--events-dir",
        str(EVENTS_DIR),
        "--arm",
        arm,
        "--grid-npz",
        str(GRID_PATH),
        "--out",
        str(field_cache_path(arm, query_env)),
    ]
    if arm == "hybrid":
        command += [
            "--train-prior-dir",
            str(prior_dirs["L0"]),
            "--query-prior-dir",
            str(prior_dirs[query_env]),
        ]
    elif arm != "gp":
        raise ValueError(f"unsupported learned arm {arm!r}")
    return command


def _fit_field(arm: str, query_env: str,
               prior_dirs: dict[str, Path]) -> dict[str, np.ndarray]:
    """Refit, validate, and load one E3 learned deployment field."""

    path = field_cache_path(arm, query_env)
    path.parent.mkdir(parents=True, exist_ok=True)
    command = gp_field_command(arm, query_env, prior_dirs)
    print(f"[e3 fields] fitting {arm}, query {query_env} -> {path.name}", flush=True)
    # Always refit.  The files are caches for provenance and inspection, not a
    # reason to trust a stale artifact created under another heading protocol.
    subprocess.run(command, check=True)
    with np.load(path) as data:
        fields = {
            camera: np.asarray(data[f"{camera}__field"], dtype=float)
            for camera in C.CAMERAS
        }
        residual_protocol = str(np.asarray(data["residual_protocol"]).reshape(-1)[0])
        if arm == "hybrid":
            expected = "fit_against_L0_prior_then_freeze"
            if residual_protocol != expected:
                raise RuntimeError(
                    f"{path}: hybrid residual protocol {residual_protocol!r}, "
                    f"expected {expected!r}"
                )
            train_prior = str(np.asarray(data["train_prior_dir"]).reshape(-1)[0])
            query_prior = str(np.asarray(data["query_prior_dir"]).reshape(-1)[0])
            if Path(train_prior).resolve() != prior_dirs["L0"].resolve():
                raise RuntimeError(f"{path}: hybrid training prior is not L0")
            if Path(query_prior).resolve() != prior_dirs[query_env].resolve():
                raise RuntimeError(f"{path}: hybrid query prior is not {query_env}")
    return fields


def _manifest(threshold: float, environments: tuple[str, ...],
              training_events: dict[str, dict[str, np.ndarray]],
              prior_dirs: dict[str, Path]) -> dict:
    field_paths = {
        "gp": field_cache_path("gp", "L0"),
        **{
            f"hybrid_{key}": field_cache_path("hybrid", key)
            for key in environments
        },
    }
    input_paths = {
        f"detector_outcomes_{key}": C.ENV_BY_KEY[key].capture / "perception_targets.csv"
        for key in environments
    }
    input_paths.update({
        f"mono_depth_{key}": C.mono_depth_path(key) for key in environments
    })
    evaluation_events = {
        key: load_evaluation_events(key, threshold) for key in environments
    }
    train_pooled = np.concatenate([
        training_events[camera]["hit"] for camera in C.CAMERAS
    ])
    evaluation_pooled = {
        key: np.concatenate([events[camera]["hit"] for camera in C.CAMERAS])
        for key, events in evaluation_events.items()
    }
    train_prevalence = float(np.mean(train_pooled))
    nominal_eval_prevalence = float(np.mean(evaluation_pooled["L0"]))
    return {
        "protocol_id": PROTOCOL_ID,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": float(threshold),
        "heading_partition": {
            "model_and_link_training": {
                "environment": "L0",
                "headings_rad": list(TRAINING_HEADINGS),
                "name": "diagonal",
            },
            "route_truth_evaluation": {
                "environments": list(environments),
                "headings_rad": list(EVALUATION_HEADINGS),
                "name": "cardinal",
            },
            "disjoint": True,
            "nominal_truth_was_not_used_for_fit_or_calibration": True,
        },
        "training_events_per_camera": {
            camera: int(len(training_events[camera]["hit"])) for camera in C.CAMERAS
        },
        "heading_partition_prevalence": {
            "descriptive_only": True,
            "training_L0_diagonal": {
                "n": int(len(train_pooled)),
                "pooled": train_prevalence,
                "per_camera": {
                    camera: float(np.mean(training_events[camera]["hit"]))
                    for camera in C.CAMERAS
                },
            },
            **{
                f"evaluation_{key}_cardinal": {
                    "n": int(len(evaluation_pooled[key])),
                    "pooled": float(np.mean(evaluation_pooled[key])),
                    "per_camera": {
                        camera: float(np.mean(evaluation_events[key][camera]["hit"]))
                        for camera in C.CAMERAS
                    },
                }
                for key in environments
            },
            "L0_train_minus_evaluation_absolute_difference": abs(
                train_prevalence - nominal_eval_prevalence
            ),
        },
        "captures": {
            key: _relative(C.ENV_BY_KEY[key].capture) for key in environments
        },
        "cache_root": _relative(WORK),
        "events_dir": _relative(EVENTS_DIR),
        "prior_dirs": {key: _relative(path) for key, path in prior_dirs.items()},
        "fields": {
            key: {"path": _relative(path), "sha256": _sha256(path)}
            for key, path in field_paths.items()
        },
        "link_calibration": {
            "protocol": (
                "pool six leave-one-spatial-block-out predictions over L0 diagonal "
                "headings, then fit one link per learned arm and camera"
            ),
            "block_x_edges": list(C.BLOCK_X_EDGES),
            "block_y_edges": list(C.BLOCK_Y_EDGES),
            "scores": {
                "path": _relative(LINK_OOF_PATH),
                "sha256": _sha256(LINK_OOF_PATH),
            },
            "generator": {
                "path": _relative(LINK_OOF_SCRIPT),
                "sha256": _sha256(LINK_OOF_SCRIPT),
            },
        },
        "inputs": {
            key: {"path": _relative(path), "sha256": _sha256(path)}
            for key, path in input_paths.items()
        },
        "gp_fitter": {
            "path": _relative(GP_FIELDS_SCRIPT),
            "sha256": _sha256(GP_FIELDS_SCRIPT),
            "hybrid_residual_protocol": (
                "fit against L0 diagonal outcomes minus the L0 monocular prior; "
                "freeze the residual; add it to each environment's query prior"
            ),
        },
    }


def build_fields(threshold: float, environments: tuple[str, ...] = ("L0", "L1")) \
        -> tuple[dict[str, dict[str, dict[str, np.ndarray]]],
                 dict[str, dict[str, np.ndarray]],
                 dict[str, dict[str, dict[str, np.ndarray]]], dict]:
    """Build E3 fields and return them with fit-only events and provenance."""

    if not environments or environments[0] != "L0" or len(set(environments)) != len(environments):
        raise ValueError("E3 environments must be unique and begin with L0")
    unknown = [key for key in environments if key not in C.ENV_BY_KEY]
    if unknown:
        raise ValueError(f"unknown E3 environments: {unknown}")

    training_events = _write_training_events(threshold)
    prior_dirs = {key: _write_prior(key) for key in environments}
    learned_link_oof = _run_link_oof(prior_dirs)
    gp = _fit_field("gp", "L0", prior_dirs)

    shared = {
        "gp": gp,
        "cad_l0": C.cad_field(C.ENV_BY_KEY["L0"].world_name),
    }
    per_environment: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for key in environments:
        arms = dict(shared)
        arms["mono_depth"] = C.mono_depth_field(key)
        arms["hybrid"] = _fit_field("hybrid", key, prior_dirs)
        arms["cad_env"] = C.cad_field(C.ENV_BY_KEY[key].world_name)
        per_environment[key] = arms

    provenance = _manifest(threshold, environments, training_events, prior_dirs)
    FIELD_MANIFEST_PATH.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return per_environment, training_events, learned_link_oof, provenance
