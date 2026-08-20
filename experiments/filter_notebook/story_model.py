"""Estimators and measurements for `camera_localisation_from_scratch`.

A separate module from `notebook_model.py` on purpose. That one is frozen evidence for
the `pp4_*` notebooks -- its grid constants and association tolerances are cited in the
defence, and changing one silently changes every number in three notebooks. Nothing here
edits it; this module imports it read-only and adds what the new story needs.

THE RUNTIME RULE, enforced by construction throughout this file:

    the filter may read   the detected pixel, the camera's mounting, wheel odometry
    the filter may NOT    ground truth, the robot's CAD model, anything fitted on the
                          drive being filtered

Functions whose name starts with `oracle_` or that take `truth_table` are SCORING
helpers. They are never called from inside an estimator.
"""

from __future__ import annotations

import math

import numpy as np

import notebook_data as nd
import notebook_model as nm
from check_route_clearance import CAMERA_XYZ, ROBOT_TOP_M, _segment_hits_box, collision_boxes

GATE_CHI2_2DOF = 5.991           # 95%, 2 degrees of freedom


# ============================================================ what the camera can see

_BOXES: list | None = None


def world_boxes() -> list:
    """Every collision box in the warehouse, cached. Read from the world file."""
    global _BOXES
    if _BOXES is None:
        _BOXES = collision_boxes()
    return _BOXES


def visibility_at(x: float, y: float) -> str:
    """What the camera can see of a robot standing at (x, y).

    Three states, and the middle one is the whole point of the occlusion section:

        clean    the camera sees where the robot meets the floor -> honest measurement
        partial  it sees the robot's top but NOT its contact point -> a detection still
                 arrives and it is DISPLACED, with nothing in the data to say so
        hidden   it sees neither -> no detection, which is harmless

    This is warehouse-reasonable: it needs the shelf layout, which a warehouse has as a
    floorplan, and the camera's mounting. It does NOT need the robot's shape -- the ray
    test uses a bare vertical segment from the floor to the robot's roof height.
    """
    boxes = world_boxes()
    cam = np.array(CAMERA_XYZ)
    ground = np.array([x, y, 0.02])
    top = np.array([x, y, ROBOT_TOP_M])
    ground_clear = not any(_segment_hits_box(cam, ground, b) for b in boxes)
    top_clear = not any(_segment_hits_box(cam, top, b) for b in boxes)
    if ground_clear and top_clear:
        return "clean"
    return "partial" if top_clear else "hidden"


# ============================================================ loading, with labels

def drive(tag: str, models=None):
    """One capture, as everything downstream wants it.

    Returns the sequence, the odometry heading, and per-detection rows carrying both the
    runtime quantities and the evaluation-only ones, clearly separated.
    """
    models = models if models is not None else nd.camera_models()
    capture = nd.load_capture(tag, models=models)
    truth_table = nd.load_truth(tag)
    seq = nm.Sequence(capture, truth_table, window=nd.route_window(tag))
    heading = nm.heading_from_odometry(seq)
    camera = models["camera_A"]

    rows = []
    for detection in capture.detections["camera_A"]:
        hit = nd.truth_at(truth_table, detection.stamp, tol_s=0.05)
        if hit is None:
            continue
        k = int(np.argmin(np.abs(seq.stamps - detection.stamp)))
        if abs(float(seq.stamps[k]) - detection.stamp) > nm.ASSOC_TOL_S:
            continue
        truth_xy = np.array([float(hit[0]), float(hit[1])])
        observed = np.asarray(detection.world, dtype=float)
        odom_yaw = float(heading[k]) if np.isfinite(heading[k]) else float("nan")
        rows.append({
            # ---- available at run time
            "step": k, "stamp": detection.stamp, "uv": (detection.u, detection.v),
            "observed": observed, "odom_yaw": odom_yaw,
            "range_m": float(np.hypot(observed[0] - camera.cam_pos[0],
                                      observed[1] - camera.cam_pos[1])),
            # ---- EVALUATION ONLY
            "truth": truth_xy, "true_yaw": float(hit[2]),
            "error": observed - truth_xy,
            "visibility": visibility_at(float(truth_xy[0]), float(truth_xy[1])),
        })
    return {"tag": tag, "capture": capture, "seq": seq, "heading": heading,
            "truth_table": truth_table, "rows": rows, "camera": camera}


def speed_of(tag: str) -> float:
    """The commanded speed of a capture, from its own manifest."""
    import json
    path = nd.capture_root(tag) / "raw" / "capture_manifest.json"
    return float(json.loads(path.read_text(encoding="utf-8")).get("speed_mps", float("nan")))


# ============================================================ is the lean constant?

def rotate(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def lean_summary(drives_out) -> list[dict]:
    """Per drive: the mean error vector, the scatter about it, and the heading."""
    out = []
    for d in drives_out:
        error = np.array([r["error"] for r in d["rows"]])
        if not len(error):
            continue
        mean = error.mean(axis=0)
        scatter = float(np.median(np.linalg.norm(error - mean, axis=1)))
        yaws = np.array([r["true_yaw"] for r in d["rows"]])
        # body-frame lean implied by this drive, if the lean rides with the robot
        body = np.array([rotate(-y) @ e for y, e in zip(yaws, error)]).mean(axis=0)
        out.append({
            "tag": d["tag"], "n": len(error),
            "mean_world_m": mean, "mean_world_cm": float(100 * np.linalg.norm(mean)),
            "scatter_cm": 100 * scatter,
            "mean_body_m": body, "mean_body_cm": float(100 * np.linalg.norm(body)),
            "median_yaw_deg": float(math.degrees(np.median(yaws))),
            "heading_span_deg": heading_span_deg(yaws),
            "median_range_m": float(np.median([r["range_m"] for r in d["rows"]])),
        })
    return out


def reversal_test(forward, backward) -> dict:
    """The same line driven both ways: does the lean stay put or ride with the robot?

    A world-frame constant predicts the two drives share a mean error. A lean carried by
    the robot's own body predicts they differ, and that the BODY-frame means agree.
    """
    out = {}
    for name, d in (("forward", forward), ("backward", backward)):
        error = np.array([r["error"] for r in d["rows"]])
        yaws = np.array([r["true_yaw"] for r in d["rows"]])
        out[name] = {
            "tag": d["tag"], "n": len(error),
            "world_m": error.mean(axis=0),
            "body_m": np.array([rotate(-y) @ e for y, e in zip(yaws, error)]).mean(axis=0),
            "yaw_deg": float(math.degrees(np.median(yaws))),
            "range_m": (float(min(r["range_m"] for r in d["rows"])),
                        float(max(r["range_m"] for r in d["rows"]))),
        }
    out["world_gap_cm"] = float(100 * np.linalg.norm(out["forward"]["world_m"]
                                                     - out["backward"]["world_m"]))
    out["body_gap_cm"] = float(100 * np.linalg.norm(out["forward"]["body_m"]
                                                    - out["backward"]["body_m"]))
    return out


# ============================================================ the mitigation ladder

def mitigation_ladder(drives_out, models) -> dict:
    """Every way of dealing with the lean, scored on the same detections.

    The rungs, and what each one costs to deploy:

        nothing            -- free
        constant (self)    -- needs ground truth ON THE DRIVE BEING FILTERED. Impossible.
                              Reported because it is the target: the best a per-drive
                              lean estimate could ever reach.
        constant (held out)-- needs ground truth once, offline. A warehouse could do this.
        geometry           -- needs the robot's CAD model registered to its odometry
                              origin. Scored with true heading and with odometry heading.
    """
    camera = models["camera_A"]
    tags = [d["tag"] for d in drives_out]
    per_tag_error = {d["tag"]: np.array([r["error"] for r in d["rows"]]) for d in drives_out}

    rows = []
    for d in drives_out:
        error = per_tag_error[d["tag"]]
        if not len(error):
            continue
        held_out = np.vstack([per_tag_error[t] for t in tags if t != d["tag"]])
        entry = {"tag": d["tag"], "n": len(error),
                 "nothing": _median_mag(error),
                 "constant_self": _median_mag(error - error.mean(axis=0)),
                 "constant_heldout": _median_mag(error - held_out.mean(axis=0))}

        for label, key in (("geometry_true", "true_yaw"), ("geometry_odom", "odom_yaw")):
            residual = []
            for r in d["rows"]:
                yaw = r[key]
                if not np.isfinite(yaw):
                    continue
                landing = nm.silhouette_bottom(camera, float(r["truth"][0]),
                                               float(r["truth"][1]), float(yaw))
                if landing is None:
                    continue
                residual.append(r["observed"] - np.asarray(landing))
            entry[label] = _median_mag(np.asarray(residual)) if residual else float("nan")
            entry[label + "_lean"] = _lean_share(np.asarray(residual)) if residual else float("nan")
        rows.append(entry)

    keys = ("nothing", "constant_self", "constant_heldout", "geometry_true", "geometry_odom")
    return {"rows": rows,
            "mean": {k: float(np.nanmean([r[k] for r in rows])) for k in keys}}


def _median_mag(v) -> float:
    v = np.asarray(v)
    return float(100 * np.median(np.linalg.norm(v, axis=1))) if len(v) else float("nan")


def _lean_share(residual) -> float:
    """What fraction of what is left is still a repeatable lean rather than scatter."""
    residual = np.asarray(residual)
    mean = np.linalg.norm(residual.mean(axis=0))
    scatter = float(np.median(np.linalg.norm(residual - residual.mean(axis=0), axis=1)))
    return float(100 * mean / (mean + scatter)) if (mean + scatter) > 0 else float("nan")


# ============================================================ the lean as a state

def bearing_to(camera, x: float, y: float) -> float:
    """Direction from the camera to a floor point. Needs only the camera's mounting."""
    return math.atan2(y - camera.cam_pos[1], x - camera.cam_pos[0])


def relative_angle(camera, x: float, y: float, yaw: float) -> float:
    """The angle between where the robot points and where the camera sees it from.

    THE variable the lean depends on. Measured 2026-08-17 across nine drives: conditioning
    the radial lean on this one number cuts its spread from 4.26 cm to 1.35 cm, where no
    fixed frame -- warehouse, robot body or sightline -- gets below 2.1 cm.

    Physically obvious once seen. The lean exists because the bottom of the detector's box
    is the bottom of the robot's OUTLINE rather than where its wheels touch, and which
    part of the robot forms that outline depends on which side the camera is looking from.
    Drive away from the camera and it sees the robot's back; drive towards it and it sees
    the front; the outline, and therefore the lean, differs.

    Available at run time: the bearing from the camera's mounting and the filter's own
    position estimate, the heading from odometry. No robot model.
    """
    return (yaw - bearing_to(camera, x, y) + math.pi) % (2 * math.pi) - math.pi


def lean_filter(seq, heading, R, camera=None, *, frame="sightline",
                sigma_lean_prior=0.10, kappa_m_per_rad=0.032, sigma_lean_walk=0.0,
                sigma_p=nm.PROCESS_SIGMA_PER_SQRT_M,
                initial_sigma=nm.INITIAL_SIGMA_M, gate=GATE_CHI2_2DOF):
    """Track the robot's position AND the camera's lean, together.

    State is [x, y, lean_1, lean_2]. What the two lean components MEAN, and how the lean
    is allowed to move, is set by `frame` -- and that choice is the whole story:

        "none"       no lean state at all. A plain filter. The baseline that fails.

        "world"      the camera leans the same way in the warehouse whichever way the
                     robot faces.                                    H = [I | I]
                     CONFOUNDED: a constant world lean and a wrong starting position
                     predict identical measurements, so only the initial-position prior
                     separates them and the estimate drifts wherever that prior puts it.

        "body"       the lean rides with the robot.          H = [I | R(theta_k)]
                     MEASURED AND REFUTED: across nine drives the body-frame lean varies
                     more between drives (3.7, 4.2 cm) than the world-frame one does
                     (2.1, 1.2 cm). Kept as a scored arm because it is the natural guess.

        "sightline"  the lean is carried in the camera's line-of-sight frame -- one
                     component along the sightline, one across it -- and is allowed to
                     MOVE as the viewing angle changes.     H = [I | B(bearing_k)]
                     This is the one the measurements support.

    The process noise on the lean is the point of the "sightline" arm. The lean is not a
    constant to be pinned down but a smooth function of the relative viewing angle, so it
    should be modelled as drifting exactly as fast as that angle changes:

        Q_lean = (kappa * |change in relative angle this step|)^2

    `kappa` is how many metres of lean a radian of viewing-angle change is worth. It is
    ONE commissioning number with a physical meaning, not a covariance floor, and the
    notebook sweeps it rather than tuning it. The angle change itself is computed from the
    filter's own estimate and odometry, so nothing here needs ground truth or a robot
    model.
    """
    dim = 2 if frame == "none" else 4
    identity = np.eye(dim)
    m = np.zeros(dim)
    m[:2] = seq.odom[0]
    P = np.zeros((dim, dim))
    P[:2, :2] = np.eye(2) * initial_sigma**2
    if dim == 4:
        P[2:, 2:] = np.eye(2) * sigma_lean_prior**2

    out = {"m": np.zeros((seq.n_steps, dim)), "P": np.zeros((seq.n_steps, dim, dim)),
           "used": np.zeros(seq.n_steps, dtype=bool),
           "rejected": np.zeros(seq.n_steps, dtype=bool),
           "nis": np.full(seq.n_steps, np.nan),
           "alpha": np.full(seq.n_steps, np.nan),
           "lean_world": np.full((seq.n_steps, 2), np.nan)}

    previous_alpha = None
    for k in range(seq.n_steps):
        u = seq.u[k]
        m[:2] = m[:2] + u
        Q = np.zeros((dim, dim))
        Q[:2, :2] = np.eye(2) * (sigma_p**2 * float(np.linalg.norm(u)))

        # how much the viewing angle moved this step, from the filter's own estimate
        alpha = float("nan")
        if camera is not None and np.isfinite(heading[k]):
            alpha = relative_angle(camera, float(m[0]), float(m[1]), float(heading[k]))
            out["alpha"][k] = alpha
        if dim == 4:
            if frame == "sightline" and np.isfinite(alpha) and previous_alpha is not None:
                step = abs((alpha - previous_alpha + math.pi) % (2 * math.pi) - math.pi)
                Q[2:, 2:] = np.eye(2) * (kappa_m_per_rad * step) ** 2
            else:
                Q[2:, 2:] = np.eye(2) * sigma_lean_walk**2
        if np.isfinite(alpha):
            previous_alpha = alpha
        P = P + Q

        if seq.camera[k] is not None:
            needs_heading = frame in ("body", "sightline")
            if needs_heading and not np.isfinite(alpha):
                # the robot is stationary, so odometry gives no heading and the lean
                # cannot be placed in the world this step. Skip rather than guess.
                out["m"][k], out["P"][k] = m, P
                continue
            H = np.zeros((2, dim))
            H[:, :2] = np.eye(2)
            if dim == 4:
                if frame == "world":
                    H[:, 2:] = np.eye(2)
                elif frame == "body":
                    H[:, 2:] = rotate(float(heading[k]))
                else:                                     # sightline
                    H[:, 2:] = rotate(bearing_to(camera, float(m[0]), float(m[1])))
                out["lean_world"][k] = H[:, 2:] @ m[2:]
            innovation = seq.y[k] - H @ m
            S = H @ P @ H.T + R
            S_inv = np.linalg.inv(S)
            nis = float(innovation @ S_inv @ innovation)
            out["nis"][k] = nis
            if nis <= gate:
                K = P @ H.T @ S_inv
                m = m + K @ innovation
                closed = identity - K @ H
                P = closed @ P @ closed.T + K @ R @ K.T
                P = 0.5 * (P + P.T)
                out["used"][k] = True
            else:
                out["rejected"][k] = True
        out["m"][k], out["P"][k] = m, P
    return out


def lean_against_angle(drives_out, models, *, bins=16) -> dict:
    """The measured lean, binned by relative viewing angle. EVALUATION ONLY.

    This is the figure that shows what the lean actually is: not a constant in any frame,
    but a smooth swing driven by one angle.
    """
    camera = models["camera_A"]
    points = []
    for d in drives_out:
        for r in d["rows"]:
            x, y = float(r["truth"][0]), float(r["truth"][1])
            beta = bearing_to(camera, x, y)
            radial = np.array([math.cos(beta), math.sin(beta)])
            across = np.array([-math.sin(beta), math.cos(beta)])
            points.append({
                "tag": d["tag"],
                "alpha_deg": math.degrees(relative_angle(camera, x, y, r["true_yaw"])),
                "radial_m": float(r["error"] @ radial),
                "across_m": float(r["error"] @ across),
                "range_m": r["range_m"], "visibility": r["visibility"],
            })
    edges = np.linspace(-180, 180, bins + 1)
    alpha = np.array([p["alpha_deg"] for p in points])
    binned = []
    for lo, hi in zip(edges, edges[1:]):
        sel = (alpha >= lo) & (alpha < hi)
        if sel.sum() < 10:
            continue
        radial = np.array([p["radial_m"] for p, s in zip(points, sel) if s])
        across = np.array([p["across_m"] for p, s in zip(points, sel) if s])
        binned.append({"centre_deg": 0.5 * (lo + hi), "n": int(sel.sum()),
                       "radial_cm": 100 * radial.mean(), "radial_sd_cm": 100 * radial.std(),
                       "across_cm": 100 * across.mean(), "across_sd_cm": 100 * across.std(),
                       "drives": len({p["tag"] for p, s in zip(points, sel) if s})})
    radial_all = np.array([p["radial_m"] for p in points])
    across_all = np.array([p["across_m"] for p in points])
    return {"points": points, "binned": binned,
            "radial_sd_cm": 100 * float(radial_all.std()),
            "across_sd_cm": 100 * float(across_all.std()),
            "radial_sd_conditioned_cm": _conditioned_sd(radial_all, alpha, bins),
            "across_sd_conditioned_cm": _conditioned_sd(across_all, alpha, bins)}


def _conditioned_sd(values, alpha, bins) -> float:
    """Spread left over once the relative viewing angle is accounted for."""
    index = np.digitize(alpha, np.linspace(-180, 180, bins + 1)) - 1
    kept = [values[index == i] - values[index == i].mean()
            for i in range(bins) if (index == i).sum() > 3]
    return 100 * float(np.concatenate(kept).std()) if kept else float("nan")


def lean_swing(drives_out) -> list[dict]:
    """How far the lean moves over a drive, and per metre driven. EVALUATION ONLY.

    The number that makes speed matter: a lean that moves 2 cm per metre driven moves ten
    times further between consecutive sightings at 1.5 m/s than at 0.15 m/s.
    """
    out = []
    for d in drives_out:
        rows = d["rows"]
        if len(rows) < 40:
            continue
        quarter = max(len(rows) // 4, 5)
        start = np.array([r["error"] for r in rows[:quarter]]).mean(axis=0)
        end = np.array([r["error"] for r in rows[-quarter:]]).mean(axis=0)
        truth = np.array([r["truth"] for r in rows])
        path = float(np.sum(np.linalg.norm(np.diff(truth, axis=0), axis=1)))
        yaws = np.array([r["true_yaw"] for r in rows])
        out.append({"tag": d["tag"], "n": len(rows), "path_m": path,
                    "start_cm": float(100 * np.linalg.norm(start)),
                    "end_cm": float(100 * np.linalg.norm(end)),
                    "swing_cm": float(100 * np.linalg.norm(end - start)),
                    "swing_per_m": float(100 * np.linalg.norm(end - start) / max(path, 1e-9)),
                    "heading_span_deg": heading_span_deg(yaws),
                    "speed_mps": speed_of(d["tag"])})
    return out


def score(result, seq, label: str, *, block=slice(0, 2)) -> dict:
    """Accuracy AND honesty together, on the position marginal. Scoring only."""
    ok = np.isfinite(seq.truth[:, 0])
    means = result["m"][:, block] if result["m"].ndim == 2 else result["m"]
    nees, errors = [], []
    for k in np.flatnonzero(ok):
        e = seq.truth[k] - means[k]
        P = result["P"][k][block, block] if result["P"].ndim == 3 else result["P"][k]
        nees.append(float(e @ np.linalg.inv(P) @ e))
        errors.append(float(np.linalg.norm(e)))
    errors = np.asarray(errors)
    nees = np.asarray(nees)
    stated = np.array([math.sqrt(max(np.trace(
        result["P"][k][block, block] if result["P"].ndim == 3 else result["P"][k]) / 2, 0.0))
        for k in np.flatnonzero(ok)])
    return {
        "label": label, "n": int(ok.sum()),
        "median_error_cm": float(100 * np.median(errors)),
        "rmse_cm": float(100 * math.sqrt((errors**2).mean())),
        "stated_sigma_cm": float(100 * np.median(stated)),
        "median_nees": float(np.median(nees)),
        "times_too_confident": float(np.median(nees) / nm.CALIBRATED_MEDIAN_NEES),
        "coverage_95": float(100 * np.mean(nees <= GATE_CHI2_2DOF)),
        "used": int(result["used"].sum()), "rejected": int(result["rejected"].sum()),
    }


# ============================================================ identifiability

def heading_span_deg(yaws) -> float:
    """How much the robot's heading varies over a drive, in degrees.

    Circular spread, so it is meaningful across the -pi/pi wrap: 1 - |mean unit vector|
    rescaled to a 0-180 degree range. A straight line scores ~0; a right-angle corner
    scores high.
    """
    yaws = np.asarray(yaws, dtype=float)
    yaws = yaws[np.isfinite(yaws)]
    if len(yaws) < 2:
        return 0.0
    resultant = np.hypot(np.mean(np.cos(yaws)), np.mean(np.sin(yaws)))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, resultant)))) * 2.0)


def bearing_span_deg(rows, camera) -> float:
    """How much the direction FROM THE CAMERA to the robot varies over a drive."""
    bearings = [math.atan2(r["truth"][1] - camera.cam_pos[1],
                           r["truth"][0] - camera.cam_pos[0]) for r in rows]
    return heading_span_deg(bearings)


def identifiability(drives_out, R, *, frame="sightline", sigma_lean_prior=0.10) -> list[dict]:
    """Recover the lean on every drive and line it up against how much the robot turned.

    The claim under test: a body-frame lean is only separable from a position error when
    the robot's heading changes during the drive. Straight drives should recover it badly
    however long they are; turning drives should recover it well however short.
    """
    out = []
    for d in drives_out:
        seq, rows = d["seq"], d["rows"]
        if len(rows) < 20:
            continue
        result = lean_filter(seq, d["heading"], R, d["camera"], frame=frame,
                             sigma_lean_prior=sigma_lean_prior)
        # EVALUATION ONLY: what the lean actually was, in the robot's frame
        yaws = np.array([r["true_yaw"] for r in rows])
        error = np.array([r["error"] for r in rows])
        true_body = np.array([rotate(-y) @ e for y, e in zip(yaws, error)]).mean(axis=0)

        final = result["m"][-1, 2:]
        lean_world = result["lean_world"][np.isfinite(result["lean_world"][:, 0])]
        posterior_sd = float(np.sqrt(np.trace(result["P"][-1][2:, 2:]) / 2))
        out.append({
            "tag": d["tag"], "n": len(rows),
            "heading_span_deg": heading_span_deg(yaws),
            "bearing_span_deg": bearing_span_deg(rows, d["camera"]),
            "recovered_body_m": final, "true_body_m": true_body,
            "recovery_error_cm": float(100 * np.linalg.norm(final - true_body)),
            "posterior_sd_cm": 100 * posterior_sd,
            "shrinkage": float(posterior_sd / sigma_lean_prior),
            **{k: v for k, v in score(result, seq, d["tag"]).items()
               if k in ("median_error_cm", "rmse_cm", "median_nees", "coverage_95")},
        })
    return out


# ============================================================ speed and occlusion

def speed_table(drives_out) -> list[dict]:
    """Sightings per metre driven, and how far the robot moves between looks."""
    out = []
    for d in drives_out:
        rows = d["rows"]
        if len(rows) < 5:
            continue
        truth = np.array([r["truth"] for r in rows])
        steps = np.linalg.norm(np.diff(truth, axis=0), axis=1)
        path = float(np.sum(np.linalg.norm(np.diff(d["seq"].truth[
            np.isfinite(d["seq"].truth[:, 0])], axis=0), axis=1)))
        out.append({
            "tag": d["tag"], "speed_mps": speed_of(d["tag"]), "n": len(rows),
            "path_m": path,
            "per_metre": len(rows) / path if path else float("nan"),
            "gap_cm": float(100 * np.median(steps)),
            "detection_rate": _detection_rate(d["tag"]),
        })
    return sorted(out, key=lambda r: (r["tag"].split("_v")[0], r["speed_mps"]))


def _detection_rate(tag: str) -> float:
    import csv
    path = nd.capture_root(tag) / "raw" / "camera_A_perception.csv"
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    return 100 * sum(1 for r in rows if r.get("detected") == "1") / max(len(rows), 1)


def occlusion_split(drives_out) -> dict:
    """Measurement error split by what the camera could actually see of the robot."""
    buckets: dict[str, list] = {"clean": [], "partial": [], "hidden": []}
    per_drive = []
    for d in drives_out:
        counts = {"clean": 0, "partial": 0, "hidden": 0}
        local: dict[str, list] = {"clean": [], "partial": [], "hidden": []}
        for r in d["rows"]:
            buckets[r["visibility"]].append(r["error"])
            local[r["visibility"]].append(r["error"])
            counts[r["visibility"]] += 1
        total = max(sum(counts.values()), 1)
        per_drive.append({
            "tag": d["tag"], "n": total,
            **{f"{k}_pct": 100 * counts[k] / total for k in counts},
            **{f"{k}_cm": _median_mag(np.asarray(local[k])) if local[k] else float("nan")
               for k in counts},
        })
    summary = {}
    for key, values in buckets.items():
        arr = np.asarray(values)
        if not len(arr):
            summary[key] = None
            continue
        mean = arr.mean(axis=0)
        summary[key] = {
            "n": len(arr), "median_cm": _median_mag(arr),
            "mean_m": mean, "mean_cm": float(100 * np.linalg.norm(mean)),
            "scatter_cm": float(100 * np.median(np.linalg.norm(arr - mean, axis=1))),
            "lean_share": _lean_share(arr),
        }
    return {"summary": summary, "per_drive": per_drive}


# ============================================================ scoring references

def oracle_R(drives_out) -> np.ndarray:
    """The scatter a perfect noise model would use. EVALUATION ONLY -- never filtered on.

    Pooled covariance of the residual about each drive's OWN mean, so it is the noise
    with every lean already removed: the most generous covariance any honest per-frame
    noise model could claim.
    """
    residuals = []
    for d in drives_out:
        error = np.array([r["error"] for r in d["rows"]])
        if len(error):
            residuals.append(error - error.mean(axis=0))
    stacked = np.vstack(residuals)
    return np.cov(stacked.T)


def detection_by_visibility(tags, models=None) -> dict:
    """How often the detector finds the robot, split by what the camera could see.

    Uses EVERY observation message, misses included -- which is the only way to see this.
    A table built from detections alone cannot show it, because the frames where the
    detector failed are exactly the ones missing from that table.
    """
    models = models if models is not None else nd.camera_models()
    pool = {k: [0, 0] for k in ("clean", "partial", "hidden")}
    per_drive = []
    for tag in tags:
        try:
            messages = nd.load_messages(tag)["camera_A"]
        except FileNotFoundError:
            continue
        truth_table = nd.load_truth(tag)
        window = nd.route_window(tag)
        counts = {k: [0, 0] for k in ("clean", "partial", "hidden")}
        for stamp, found in messages:
            if window and not (window[0] <= stamp <= window[1]):
                continue
            hit = nd.truth_at(truth_table, stamp, tol_s=0.05)
            if hit is None:
                continue
            key = visibility_at(float(hit[0]), float(hit[1]))
            counts[key][0] += 1
            counts[key][1] += int(found)
            pool[key][0] += 1
            pool[key][1] += int(found)
        per_drive.append({"tag": tag, **{
            k: {"chances": counts[k][0], "found": counts[k][1],
                "rate": 100 * counts[k][1] / counts[k][0] if counts[k][0] else float("nan")}
            for k in counts}})
    return {"pooled": {k: {"chances": v[0], "found": v[1],
                           "rate": 100 * v[1] / v[0] if v[0] else float("nan")}
                       for k, v in pool.items()},
            "per_drive": per_drive}


# ============================================================ the system

class MultiSequence:
    """A state sequence that keeps EVERY camera's sighting, on a phase-aligned grid.

    Three things `notebook_model.Sequence` does that this does not, each of which was
    measured to matter on 2026-08-17:

    1. IT KEEPS SIMULTANEOUS SIGHTINGS. `Sequence` drops any second camera reporting at
       the same step ("one observation per step, as the model assumes"). In the four-camera
       capture that silently discards 277 simultaneous sightings across all six camera
       pairs -- which is exactly the data that measures one camera's lean against another's
       without any ground truth. Fusion cannot be studied on a loader that throws fusion
       data away.

    2. IT PHASE-ALIGNS THE GRID. Detections land on exact multiples of 0.1 s while
       `Sequence` starts its grid at an arbitrary time, so every detection is snapped by a
       CONSTANT 12-36 ms. At 0.15 m/s that is 0.2-0.5 cm and harmless; at 1.5 m/s it is
       2.7 cm, a third of the effect under study, and it masquerades as a speed-dependent
       lean. Quantising the grid origin drives it to zero. The frozen `pp4_*` notebooks
       keep the old behaviour; this is a separate class so their numbers do not move.

    3. IT TAKES A KNOWN START POSE. A warehouse robot that has just picked a package from
       a known station knows where it is to a couple of centimetres. That prior is the
       DOMINANT term in the position uncertainty -- measured, the camera contributes
       nothing to the absolute position, so the start pose is what sets it.

    The measurement is a pixel and a camera. Nothing here cares whether that pixel is a
    detector's box bottom or a learned ground-contact keypoint, so a keypoint detector
    drops in without touching this class.
    """

    def __init__(self, capture, truth_table, *, grid_hz=nm.GRID_HZ, window=None,
                 assoc_tol_s=nm.ASSOC_TOL_S, align_grid=True, start_xy=None):
        stamps = np.asarray(capture.stamps, dtype=float)
        odom = np.asarray(capture.odom, dtype=float)
        lo = stamps[0] if window is None else max(stamps[0], window[0])
        hi = stamps[-1] if window is None else min(stamps[-1], window[1])
        step = 1.0 / grid_hz
        if align_grid:
            lo = math.floor(lo / step) * step        # <- the whole snap fix
        grid = np.arange(lo, hi, step)

        self.stamps = grid
        self.dt = step
        self.odom = np.column_stack([np.interp(grid, stamps, odom[:, i]) for i in range(2)])
        self.u = np.vstack([np.zeros((1, 2)), np.diff(self.odom, axis=0)])
        self.cameras = tuple(capture.cameras)
        self.start_xy = (np.asarray(start_xy, dtype=float) if start_xy is not None
                         else self.odom[0].copy())

        # every sighting at every step, not just the first
        self.sightings: list[list[dict]] = [[] for _ in grid]
        self.snap_s: list[float] = []
        for camera in self.cameras:
            for detection in capture.detections[camera]:
                k = int(np.argmin(np.abs(grid - detection.stamp)))
                gap = abs(float(grid[k]) - detection.stamp)
                if gap > assoc_tol_s:
                    continue
                self.snap_s.append(gap)
                self.sightings[k].append({
                    "camera": camera, "uv": (detection.u, detection.v),
                    "world": np.asarray(detection.world, dtype=float),
                    "stamp": detection.stamp,
                })

        # EVALUATION ONLY
        self.truth = np.full((len(grid), 2), np.nan)
        self.truth_yaw = np.full(len(grid), np.nan)
        for k, stamp in enumerate(grid):
            hit = nd.truth_at(truth_table, float(stamp))
            if hit is not None:
                self.truth[k] = hit[:2]
                self.truth_yaw[k] = hit[2]

    @property
    def n_steps(self) -> int:
        return len(self.stamps)

    @property
    def observed(self) -> np.ndarray:
        return np.array([bool(s) for s in self.sightings])

    def simultaneous(self, min_cameras=2):
        """Steps where two or more cameras saw the robot at once -- the handover data."""
        return [k for k, s in enumerate(self.sightings)
                if len({x["camera"] for x in s}) >= min_cameras]

    def snap_report(self) -> dict:
        snap = np.asarray(self.snap_s)
        if not len(snap):
            return {"n": 0}
        return {"n": len(snap), "median_ms": 1000 * float(np.median(snap)),
                "p95_ms": 1000 * float(np.percentile(snap, 95)),
                "max_ms": 1000 * float(snap.max())}


# ---- the two-camera DEVELOPMENT world.
#
# Registered here rather than in `notebook_data.py`, which is frozen evidence for the
# `pp4_*` notebooks. Adding a world there would be harmless, but "harmless" is what the
# audit brief warns about, so this stays additive and local.
#
# The detector is `warehouse_yolo_detector_v1` -- the same weights as the single-camera
# world, captured and trained in this warehouse. Camera B sees the same warehouse from a
# different wall, so those weights apply; the four-camera world's v3 weights do NOT, they
# were trained on the flagship world's viewpoints.
AWS_2CAM = nd.World(
    key="warehouse_aws_2cam",
    world_sdf_name="warehouse_aws_2cam.world.sdf",
    cameras=("camera_A", "camera_B"),
    model_includes={"camera_A": "external_camera", "camera_B": "external_camera_b"},
    image_topics={"camera_A": "/external_camera/image_raw",
                  "camera_B": "/external_camera_b/image_raw"},
    commissioned_file="commissioned_observation_noise_aws.json",
    detector_model="warehouse_yolo_detector_v1",
    description="AWS warehouse, south-wall + east-wall cameras, for fusion development",
)
nd.WORLDS.setdefault(AWS_2CAM.key, AWS_2CAM)
