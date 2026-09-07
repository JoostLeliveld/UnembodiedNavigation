"""Offline evidence loading, identity validation and own-time reference scoring.

Camera readings use obs_stamp, fused corrections use fused_stamp, and public
beliefs use planner_belief_stamp. Committed posteriors require explicit records.
A finite interpolation bound must be supplied by the selected analysis protocol;
without one, only exact logged ground-truth timestamps have reference support.
Schema <4 capture-stamp fallbacks are historical diagnostics only. Ground truth
is an offline scoring reference and never an estimator input.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src/unav_common"))
from unav_common.correction_ledger import validate_correction_ledger

class PosteriorUnavailable(ValueError):
    """The log does not contain an identifiable committed posterior."""


class TruthSeries:
    """The run's ground-truth path, on ground truth's own clock."""

    def __init__(self, t, x, y, yaw, source: str, *, max_reference_gap_s=None):
        t, x, y, yaw = (np.asarray(v, dtype=float) for v in (t, x, y, yaw))
        if not (t.ndim == x.ndim == y.ndim == yaw.ndim == 1 and
                len(t) == len(x) == len(y) == len(yaw) and len(t)):
            raise ValueError("ground truth must contain equally sized nonempty vectors")
        if not np.isfinite(np.c_[t, x, y]).all():
            raise ValueError("nonfinite ground truth time/position")
        if np.any(np.diff(t) < 0):
            raise ValueError("ground truth clock reset or reordered reference samples")
        keep = np.r_[True, np.diff(t) != 0]
        for i in np.flatnonzero(~keep):
            a, b = np.array([x[i-1], y[i-1], yaw[i-1]]), np.array([x[i], y[i], yaw[i]])
            if not np.array_equal(a, b, equal_nan=True):
                raise ValueError("conflicting ground truth samples at one timestamp")
        self.t, self.x, self.y, self.yaw = (v[keep] for v in (t, x, y, yaw))
        # Unwrap separately across missing heading sections; a missing yaw must not
        # poison later valid positions/headings or license interpolation across it.
        indices = np.flatnonzero(np.isfinite(self.yaw))
        for block in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
            self.yaw[block] = np.unwrap(self.yaw[block])
        if max_reference_gap_s is not None and (not math.isfinite(max_reference_gap_s) or
                                                max_reference_gap_s <= 0):
            raise ValueError("max_reference_gap_s must be finite and positive")
        self.max_gap_s = max_reference_gap_s
        self.source = source

    def __len__(self):
        return int(self.t.size)

    def support(self, stamps):
        """Exact samples or bounded interpolation brackets; no extrapolation."""
        s = np.asarray(stamps, dtype=float)
        right = np.searchsorted(self.t, s, side="left")
        r = np.clip(right, 0, len(self.t)-1)
        left = np.maximum(r-1, 0)
        exact = s == self.t[r]
        inside = (right > 0) & (right < len(self.t))
        bracket = self.t[r] - self.t[left]
        supported = exact if self.max_gap_s is None else exact | (inside & (bracket <= self.max_gap_s + 1e-9))
        return np.isfinite(s) & supported

    def at(self, stamps):
        """Truth at `stamps`. NaN outside the recorded interval -- never clamped.

        Clamping to an endpoint is how a misaligned comparison hides: it returns a real
        pose from the wrong instant and every downstream statistic accepts it.
        """
        s = np.asarray(stamps, dtype=float)
        gx = np.interp(s, self.t, self.x)
        gy = np.interp(s, self.t, self.y)
        outside = ~self.support(s)
        gx = np.where(outside, np.nan, gx)
        gy = np.where(outside, np.nan, gy)
        return gx, gy

    def yaw_at(self, stamps):
        s = np.asarray(stamps, dtype=float)
        gyaw = np.interp(s, self.t, self.yaw)
        outside = ~self.support(s)
        return np.where(outside, np.nan, gyaw)


def schema_version(run: Path) -> int:
    """Which logging conventions this run was written under."""

    manifest = Path(run) / "run_manifest.json"
    if not manifest.is_file():
        return 1
    try:
        record = json.loads(manifest.read_text())
        version = record["logging_schema_version"]
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError("logging_schema_version must be a positive integer")
        return version
    except (ValueError, TypeError, KeyError, OSError) as exc:
        raise ValueError(f"{manifest}: invalid logging schema") from exc


def _float(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def rows(run: Path) -> list[dict]:
    with (Path(run) / "experiment.csv").open() as stream:
        table = list(csv.DictReader(stream))
    _validate_log_clock(table)
    return table


def _validate_log_clock(table):
    stamps = np.array([_float(r, "stamp") for r in table])
    if not np.isfinite(stamps).all() or np.any(np.diff(stamps) < 0):
        raise ValueError("invalid logger clock or unmarked clock reset")


def truth_series(run: Path, table: list[dict] | None = None, *, max_reference_gap_s=None) -> TruthSeries:
    """The truth path, timestamped as well as the run allows.

    Schema 2 logs `gt_stamp`, the stamp the pose itself carried, so the series sits on
    ground truth's own clock. Schema 1 has only the log clock, which is late by however
    long the held value had been sitting there -- bounded by the 10 Hz log tick.
    """
    table = rows(run) if table is None else table
    _validate_log_clock(table)
    t, x, y, yaw = [], [], [], []
    use_gt_stamp = schema_version(run) >= 2 and table and "gt_stamp" in table[0]
    for row in table:
        if _float(row, "gt_available") != 1.0:
            continue
        stamp = _float(row, "gt_stamp") if use_gt_stamp else _float(row, "stamp")
        gx, gy = _float(row, "gt_x"), _float(row, "gt_y")
        if not (math.isfinite(stamp) and math.isfinite(gx) and math.isfinite(gy)):
            continue
        t.append(stamp)
        x.append(gx)
        y.append(gy)
        yaw.append(_float(row, "gt_yaw"))
    if not t:
        raise SystemExit(f"{run}: no usable ground truth")
    summary_path = Path(run) / "run_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    provenance = summary.get("gt_stamp_source", "unspecified")
    return TruthSeries(t, x, y, yaw,
                       f"gt_stamp ({provenance})" if use_gt_stamp else "log_stamp (schema 1)",
                       max_reference_gap_s=max_reference_gap_s)


def aligned_error_cm(run: Path, kind: str, table: list[dict] | None = None, *, max_reference_gap_s=None) -> dict:
    """Error of one estimate against the truth at the estimate's OWN stamp.

    `kind` is 'belief' (the planner's belief) or 'state' (the correction the filter is
    holding). Returns centimetres, plus the same quantity scored at log time so the size
    of the old artefact stays visible instead of being asserted.
    """
    if kind not in {"belief", "state"}:
        raise ValueError("kind must be 'belief' or 'state'")
    if schema_version(run) >= 4:
        validate_run_ledger(run)
    table = rows(run) if table is None else table
    truth = truth_series(run, table, max_reference_gap_s=max_reference_gap_s)
    xkey, ykey, skey = (
        ("planner_belief_x", "planner_belief_y", "planner_belief_stamp")
        if kind == "belief" else ("state_x", "state_y", "state_stamp"))

    est_x = np.array([_float(r, xkey) for r in table])
    est_y = np.array([_float(r, ykey) for r in table])
    est_s = np.array([_float(r, skey) for r in table])
    log_s = np.array([_float(r, "stamp") for r in table])

    have = np.isfinite(est_x) & np.isfinite(est_y) & np.isfinite(est_s)
    available_key = "planner_belief_available" if kind == "belief" else "state_available"
    if table and available_key in table[0]:
        have &= np.array([_float(r, available_key) == 1 for r in table])
    if schema_version(run) >= 4:
        seen = {}
        payload_keys = ([xkey, ykey, "planner_belief_yaw", "planner_cov_x", "planner_cov_xy", "planner_cov_y"]
                        if kind == "belief" else [xkey, ykey])
        for row, stamp, present in zip(table, est_s, have):
            if not present:
                continue
            payload = np.array([_float(row, k) for k in payload_keys])
            if stamp in seen and not np.array_equal(seen[stamp], payload, equal_nan=True):
                raise ValueError(f"{run}: ambiguous {kind} payloads at one timestamp without message revision")
            seen[stamp] = payload
    gx_a, gy_a = truth.at(est_s)
    gx_l, gy_l = truth.at(log_s)
    aligned = np.where(have, np.hypot(est_x - gx_a, est_y - gy_a), np.nan) * 100.0
    logtime = np.where(have, np.hypot(est_x - gx_l, est_y - gy_l), np.nan) * 100.0
    lag = np.where(have, log_s - est_s, np.nan)
    return dict(aligned_cm=aligned, logtime_cm=logtime, stamp=est_s, log_stamp=log_s,
                x=est_x, y=est_y, gt_x=gx_a, gt_y=gy_a, have=have, lag_s=lag,
                truth_source=truth.source, reference_supported=truth.support(est_s),
                reference_max_gap_s=truth.max_gap_s,
                reference_method=("exact_logged_gt" if truth.max_gap_s is None
                                  else "bounded_interpolation_of_logged_gt"))


def landed_mask(stamps) -> np.ndarray:
    """True on the first row that carries each distinct message stamp.

    A held message is re-logged every tick, and re-scoring it against a robot that has
    moved on measures the robot's travel, not the sensor.
    """
    s = np.asarray(stamps, dtype=float)
    if s.size == 0:
        return np.zeros(0, dtype=bool)
    seen = set()
    first = np.zeros(s.size, dtype=bool)
    for i, stamp in enumerate(s):
        if np.isfinite(stamp) and stamp not in seen:
            first[i] = True
            seen.add(stamp)
    return first


def corrections(run: Path, table: list[dict] | None = None) -> dict:
    """When a fresh correction actually landed, and how long the gaps were.

    `n_state_publications` is what the old `corrections` field counted: log rows whose
    `state_stamp` changed. The manager publishes at 20 Hz and the logger samples at
    10 Hz, so that number is neither the corrections published nor the camera readings
    behind them -- it is the fraction of the log that had a fresh correction, times the
    duration. `n_detector_rounds` is the honest count where the run logs capture times.
    """
    table = rows(run) if table is None else table
    state_s = np.array([_float(r, "state_stamp") for r in table])
    available = np.array([_float(r, "state_available") == 1.0 for r in table])
    unique = np.array(sorted({float(s) for s, ok in zip(state_s, available)
                              if ok and math.isfinite(s)}))
    gaps = np.diff(unique) if unique.size > 1 else np.array([])
    log_s = np.array([_float(r, "stamp") for r in table])
    finite_log = log_s[np.isfinite(log_s)]
    duration_s = float(finite_log[-1] - finite_log[0]) if finite_log.size > 1 else math.nan

    n_detector_rounds = None
    obs = observations(run)
    if obs:
        source_batches = {o["source_batch_id"] for o in obs if o["source_batch_id"]}
        if source_batches:
            n_detector_rounds = len(source_batches)
        else:
            # Legacy fallback: a tuple of the camera capture stamps identifies one
            # detector round more honestly than counting each camera as a round.
            by_decision = {}
            for item in obs:
                decision = round(item["decision_stamp"], 6)
                if math.isfinite(item["obs_stamp"]):
                    by_decision.setdefault(decision, set()).add(
                        round(item["obs_stamp"], 6))
            n_detector_rounds = len({
                tuple(sorted(stamps)) for stamps in by_decision.values() if stamps
            }) or None

    result = dict(
        n_state_publications=int(unique.size),
        n_state_publications_note=(
            "log rows with a fresh /state/bev stamp, at the 10 Hz log rate -- an "
            "availability fraction, not a count of corrections"),
        n_detector_rounds=n_detector_rounds,
        duration_s=duration_s,
        state_fresh_rate_hz=(float(unique.size) / duration_s
                             if duration_s and math.isfinite(duration_s) else math.nan),
        longest_gap_s=float(gaps.max()) if gaps.size else math.nan,
        median_gap_s=float(np.median(gaps)) if gaps.size else math.nan,
    )

    if schema_version(run) >= 4:
        accounting = correction_accounting(run)
        result.update(longest_gap_s=accounting['longest_correction_gap_s'],
                      median_gap_s=accounting['median_correction_gap_s'],
                      accepted_updates=accounting['accepted_updates'],
                      correction_dropped_fraction=accounting['correction_dropped_fraction'],
                      gap_reference=accounting['gap_reference'])
    else:
        result['gap_reference'] = 'legacy held-state diagnostic; accepted-update gaps unavailable'
    return result


def observations(run: Path) -> list[dict]:
    """Raw `fusion_observations.csv`, one dict per row, floats parsed. No filtering."""

    path = Path(run) / "fusion_observations.csv"
    if not path.is_file():
        return []
    out = []
    for row in csv.DictReader(open(path)):
        entry = dict(
            camera=row.get("camera", ""),
            source_batch_id=row.get("source_batch_id", "").strip(),
            used=row.get("used") == "1",
            decision_stamp=_float(row, "stamp"),
            common_capture_stamp=_float(row, "common_capture_stamp"),
            obs_stamp=_float(row, "obs_stamp"),
            obs_x=_float(row, "obs_x"),
            obs_y=_float(row, "obs_y"),
            fused_x=_float(row, "fused_x"),
            fused_y=_float(row, "fused_y"),
            fused_stamp=_float(row, "fused_stamp"),
            n_candidates=_float(row, "n_candidates"),
            n_used=_float(row, "n_used"),
            conf=_float(row, "conf"),
            bbox_h_px=_float(row, "bbox_h_px"),
            bbox_w_px=_float(row, "bbox_w_px"),
            # schema 5 onward: the box the hull model predicted from the pose the
            # correction was made from. NaN on schema <= 4 drives, which never logged it.
            pred_h_px=_float(row, "pred_h_px"),
            pred_w_px=_float(row, "pred_w_px"),
            range_m=_float(row, "range_m"),
            obs_repeat=_float(row, "obs_repeat"),
        )
        entry["cov"] = np.array([
            [_float(row, "obs_cov_xx"), _float(row, "obs_cov_xy")],
            [_float(row, "obs_cov_xy"), _float(row, "obs_cov_yy")]])
        entry["fused_cov"] = np.array([
            [_float(row, "fused_cov_xx"), _float(row, "fused_cov_xy")],
            [_float(row, "fused_cov_xy"), _float(row, "fused_cov_yy")]])
        entry["aligned_xy"] = np.array([
            _float(row, "aligned_x"), _float(row, "aligned_y")])
        entry["aligned_cov"] = np.array([
            [_float(row, "aligned_cov_xx"), _float(row, "aligned_cov_xy")],
            [_float(row, "aligned_cov_xy"), _float(row, "aligned_cov_yy")]])
        out.append(entry)
    return out


def assimilations(run: Path) -> list[dict]:
    """Terminal recursive-filter outcome keyed by physical detector batch."""
    path = Path(run) / "correction_assimilations.csv"
    if not path.is_file():
        return []
    out = []
    seen = set()
    for row in csv.DictReader(open(path)):
        source_batch_id = str(row.get("source_batch_id", "") or "").strip()
        if not source_batch_id:
            raise ValueError(f"{path}: assimilation without source_batch_id")
        if source_batch_id in seen:
            raise ValueError(f"{path}: duplicate assimilation for {source_batch_id}")
        seen.add(source_batch_id)
        accepted = str(row.get("accepted", "")).strip()
        if accepted not in {"0", "1"}:
            raise ValueError(f"{path}: invalid accepted flag for {source_batch_id}")
        out.append({
            **row,
            "source_batch_id": source_batch_id,
            "correction_stamp": _float(row, "correction_stamp"),
            "apply_stamp": _float(row, "apply_stamp"),
            "belief_stamp_after": _float(row, "belief_stamp_after"),
            "status": str(row.get("status", "") or "").strip(),
            "reason": str(row.get("reason", "") or "").strip(),
            "accepted": accepted == "1",
            "nis": _float(row, "nis"),
        })
        for name in ("correction_stamp_ns", "apply_stamp_ns", "belief_stamp_before_ns", "belief_stamp_after_ns"):
            if row.get(name) not in (None, ""):
                out[-1][name] = _integer(row[name], name)
            else:
                out[-1].pop(name, None)
    return out


def validate_run_ledger(run: Path):
    """Validate the complete raw publication/outcome ledger, before truth selection.

    A valid event beyond the final truth row remains accounted for. This validator
    makes no accuracy claim and never uses ``fused_answers`` as its input population.
    """
    run = Path(run)
    if schema_version(run) < 4:
        raise ValueError(f"{run}: schema 4 source-batch accounting is required")
    for name in ("fusion_observations.csv", "correction_assimilations.csv"):
        if not (run / name).is_file():
            raise ValueError(f"{run}: missing {name}")
    outcomes = []
    for a in assimilations(run):
        canonical = dict(a)
        canonical.pop("epoch", None)  # Belief epoch is not the detector/source epoch.
        if a.get("source_epoch"):
            canonical["epoch"] = a["source_epoch"]
        outcomes.append(canonical)
    if schema_version(run) >= 8:
        path = run / "correction_publications.csv"
        if not path.is_file():
            raise ValueError(f"{run}: schema 8 publication ledger is missing")
        with path.open() as stream:
            publications = list(csv.DictReader(stream))
        for p in publications:
            p["member_ids"] = json.loads(p.get("member_ids") or p["accepted_camera_ids"])
            for name in ("correction_stamp_ns", "common_capture_stamp_ns", "publication_stamp_ns"):
                if p.get(name) not in (None, ""):
                    p[name] = _integer(p[name], name)
                else:
                    p.pop(name, None)
            if not p.get("frame_id") or str(p.get("schema_version")) not in {"1", "2"}:
                raise ValueError(f"{run}: malformed correction publication")
            if str(p["schema_version"]) == "2":
                matches = [a for a in outcomes if a["source_batch_id"] == p["source_batch_id"]]
                if not p.get("epoch") or any(not a.get("source_epoch") for a in matches):
                    raise ValueError(f"{run}: source epoch required for v2 correction lineage")
                if "correction_stamp_ns" not in p or any("correction_stamp_ns" not in a or
                                                        "apply_stamp_ns" not in a for a in matches):
                    raise ValueError(f"{run}: exact correction/apply timestamps required for v2 lineage")
        ledger = validate_correction_ledger(publications, outcomes).require_valid()
    else:
        ledger = None
    publications = []
    for o in observations(run):
        publications.append(dict(source_batch_id=o["source_batch_id"],
            correction_stamp=o["fused_stamp"], payload=dict(
                mean=[o["fused_x"], o["fused_y"]], covariance=o["fused_cov"].tolist())))
    if ledger is not None:
        diagnostic_ids = {p["source_batch_id"] for p in publications}
        if not diagnostic_ids <= ledger.publications_by_batch.keys():
            raise ValueError(f"{run}: fusion diagnostic without a correction publication")
        for p in publications:
            stamp = ledger.publications_by_batch[p["source_batch_id"]]["correction_stamp"]
            if not math.isclose(float(stamp), p["correction_stamp"], abs_tol=1e-9, rel_tol=0):
                raise ValueError(f"{run}: fusion diagnostic/publication timestamp mismatch")
        outcomes = [a for a in outcomes if a["source_batch_id"] in diagnostic_ids]
    result = validate_correction_ledger(publications, outcomes,
                                       allow_repeated_publications=True)
    result.require_valid()
    return ledger if ledger is not None else result


def _unique_observations(run, obs):
    """One camera per physical batch; inconsistent copies are evidence errors."""
    if schema_version(run) < 4:
        return obs  # Explicit diagnostic legacy capture-stamp handling below.
    seen = {}
    for o in obs:
        key = o["source_batch_id"], o["camera"]
        if not all(key):
            raise ValueError(f"{run}: missing camera/source_batch_id")
        if key in seen:
            previous = seen[key]
            for field in ("obs_stamp", "obs_x", "obs_y", "cov", "used", "conf",
                          "bbox_h_px", "bbox_w_px", "aligned_xy", "aligned_cov"):
                if not np.array_equal(np.asarray(o[field]), np.asarray(previous[field]), equal_nan=True):
                    raise ValueError(f"{run}: conflicting camera reading {key}: {field}")
        else:
            seen[key] = o
    return list(seen.values())


def mission_interval(run: Path, table=None):
    """A declared whole mission interval; do not omit either blind endpoint."""
    path = Path(run) / "run_summary.json"
    summary = json.loads(path.read_text()) if path.is_file() else {}
    start, stop = _float(summary, "first_cmd_stamp"), _float(summary, "stop_stamp")
    if not np.isfinite([start, stop]).all() or stop < start:
        raise ValueError(f"{run}: missing or invalid mission interval")
    return start, stop


def verify_frozen_entry(entry, required, *, minimum_schema=4, repo=REPO):
    """Verify every required artifact and the declared run/task/seed identity."""
    run = (Path(repo) / entry["run"]).resolve()
    files = entry.get("files", {})
    missing = set(required) - files.keys()
    if missing:
        raise ValueError(f"{run}: frozen artifact hashes missing: {sorted(missing)}")
    for name, expected in files.items():
        path = (run / name).resolve()
        if not path.is_relative_to(run) or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"{run}: invalid frozen artifact entry {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"{run}: changed frozen artifact {name}")
    manifest = json.loads((run / "run_manifest.json").read_text())
    if schema_version(run) < minimum_schema:
        raise ValueError(f"{run}: schema {minimum_schema} or newer required")
    if schema_version(run) >= 8 and not {"correction_publications.csv", "runtime_event_deliveries.jsonl"} <= files.keys():
        raise ValueError(f"{run}: schema 8 frozen event ledgers are required")
    if "task" not in entry or "seed" not in entry:
        raise ValueError("frozen task and seed are required")
    if manifest.get("task") != entry["task"] or manifest.get("seed") != entry["seed"]:
        raise ValueError(f"{run}: frozen task/seed identity mismatch")
    return run, manifest, json.loads((run / "run_summary.json").read_text())


def required_artifacts(run, base):
    names = list(base)
    if schema_version(run) >= 8:
        names.extend(["correction_publications.csv", "runtime_event_deliveries.jsonl"])
    return tuple(dict.fromkeys(names))


def measured_odometry(table, *, start=None, stop=None):
    """Finite ZOH controls, with duplicate agreement and chronological support."""
    odom = {}
    previous = -math.inf
    for row in table:
        stamp = _float(row, "odom_noisy_stamp")
        control = np.array([_float(row, "odom_noisy_v"), _float(row, "odom_noisy_w")])
        if not math.isfinite(stamp):
            continue  # The logger may not yet have received odometry.
        if not np.isfinite(control).all():
            raise ValueError("nonfinite measured odometry")
        if stamp < previous:
            raise ValueError("reordered odometry or unmarked clock reset")
        previous = stamp
        if stamp in odom and not np.array_equal(odom[stamp], control):
            raise ValueError("conflicting odometry controls at one timestamp")
        odom[stamp] = control
    if start is not None:
        if not any(t <= start for t in odom):
            raise ValueError("no causal measured control at mission start")
        anchor = max(t for t in odom if t <= start)
        odom = {t: u for t, u in odom.items() if t >= start} | {start: odom[anchor]}
    if stop is not None:
        odom = {t: u for t, u in odom.items() if t <= stop}
    if not odom:
        raise ValueError("no measured odometry")
    return dict(sorted(odom.items()))


def camera_opportunities(run):
    """Canonical detector deliveries, verified before time/reference filtering."""
    deliveries = [json.loads(line) for line in (Path(run) / "camera_opportunities.jsonl").read_text().splitlines()]
    groups = {}
    duplicates = 0
    for row in deliveries:
        if row.get("valid_contract") is not True or row.get("conflicting_duplicate"):
            raise ValueError("invalid or conflicting camera opportunity delivery")
        o = row["observation"]
        key = o.get("source_batch_id"), o.get("camera_id")
        if not all(isinstance(v, str) and v for v in key):
            raise ValueError("camera opportunity identity missing")
        if not math.isfinite(float(o["timestamp_s"])) or not isinstance(o["detection_valid"], bool):
            raise ValueError("invalid camera opportunity time/detection flag")
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in groups.items():
        canonical = [r for r in group if r.get("duplicate") is False]
        if len(canonical) != 1:
            raise ValueError(f"camera opportunity needs one canonical delivery: {key}")
        o = canonical[0]["observation"]
        for row in group:
            if row["observation"] != o:
                raise ValueError(f"conflicting camera opportunity copies: {key}")
        duplicates += len(group)-1
        result.append(o)
    return sorted(result, key=lambda o: (o["timestamp_s"], o["source_batch_id"], o["camera_id"])), duplicates


def freeze_analysis_protocol(out, name, protocol, *, owned_outputs=()):
    """Do not replace a historical result with different or unidentified analysis."""
    out = Path(out)
    path = out / name
    if path.is_file():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f"{path}: analysis inputs differ; choose a new output directory")
    elif any((out / p).exists() for p in owned_outputs):
        raise ValueError(f"{out}: existing outputs lack matching analysis provenance; preserve them")
    else:
        out.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
    return protocol


def correction_accounting(run: Path):
    ledger = validate_run_ledger(run)
    start, stop = mission_interval(run)
    outcomes = [a for a in ledger.by_batch.values() if start <= a["apply_stamp"] <= stop]
    accepted = [a["apply_stamp"] for a in outcomes if a["source_batch_id"] in ledger.accepted_update_ids]
    gaps = np.diff(sorted({start, stop, *accepted}))
    return dict(total_published=len(ledger.publications_by_batch), total_outcomes=len(ledger.by_batch),
                mission_outcomes=len(outcomes), accepted_updates=len(accepted),
                rejected=sum(a["status"] == "rejected" for a in outcomes),
                dropped=sum(a["status"] == "dropped" for a in outcomes),
                correction_dropped_fraction=(sum(a["status"] == "dropped" for a in outcomes) /
                                             len(outcomes) if outcomes else None),
                longest_correction_gap_s=float(max(gaps, default=0.0)),
                median_correction_gap_s=float(np.median(gaps)) if gaps.size else 0.0,
                gap_reference="accepted apply_stamp within [first_cmd_stamp, stop_stamp], including endpoints")


def readings(run: Path, *, admitted_only: bool = True, dedupe: bool = True,
             require_capture_time: bool = True, max_reference_gap_s=None,
             require_reference: bool = True) -> list[dict]:
    """One entry per camera reading, scored against the truth when the camera saw it.

    `dedupe` keeps the FIRST row for each (camera, capture time): the reading as the
    manager first computed it. Without it every statistic below counts each reading
    about four times and weights cameras by how long the manager re-fused them.

    `admitted_only` restricts to readings the arm's rule used. That is what the runtime
    consumed, but it conditions on passing the disagreement gate -- on agreeing with the
    other cameras -- so it is not a clean per-camera error. Pass False for the
    unconditional distribution; callers that care should report both.
    """
    obs = observations(run)
    if schema_version(run) >= 4:
        if not dedupe:
            raise ValueError("modern reading scores require deduplication; observations() exposes raw rows")
        validate_run_ledger(run)
        obs = _unique_observations(run, obs)
    if not obs:
        return []
    have_capture = any(math.isfinite(o["obs_stamp"]) for o in obs)
    if require_capture_time and not have_capture:
        return []
    truth = truth_series(run, max_reference_gap_s=max_reference_gap_s)

    seen = set()
    out = []
    for o in obs:
        if admitted_only and not o["used"]:
            continue
        if not (math.isfinite(o["obs_x"]) and math.isfinite(o["obs_y"])):
            continue
        cap = o["obs_stamp"]
        if not math.isfinite(cap):
            continue
        if dedupe:
            key = ((o["source_batch_id"], o["camera"]) if schema_version(run) >= 4
                   else (o["camera"], cap))
            if key in seen:
                continue
            seen.add(key)
        gx, gy = truth.at([cap])
        supported = math.isfinite(gx[0]) and math.isfinite(gy[0])
        if require_reference and not supported:
            continue
        entry = dict(o)
        entry["reference_supported"] = supported
        entry["truth"] = np.array([float(gx[0]), float(gy[0])])
        entry["truth_yaw"] = float(truth.yaw_at([cap])[0])
        entry["error"] = np.array([o["obs_x"], o["obs_y"]]) - entry["truth"]
        entry["error_cm"] = float(np.linalg.norm(entry["error"]) * 100.0)
        out.append(entry)
    out.sort(key=lambda r: (r["obs_stamp"], r["source_batch_id"], r["camera"]))
    return out


def fused_answers(run: Path, *, dedupe: bool = True, max_reference_gap_s=None) -> list[dict]:
    """The fused correction per decision, scored at the instant IT describes.

    The manager propagates the fused answer to `now` and re-stamps it, while leaving
    each camera's reading at capture time. Scoring both against one truth -- as the
    fusion-quality check used to -- charges every individual camera ~200 ms of robot
    travel and charges the fused answer none, which is a comparison the fusion rule wins
    by convention.
    """
    obs = observations(run)
    if schema_version(run) >= 4:
        if not dedupe:
            raise ValueError("modern fused scores require one event per identity")
        validate_run_ledger(run)
        obs = _unique_observations(run, obs)
    if not obs:
        return []
    truth = truth_series(run, max_reference_gap_s=max_reference_gap_s)
    by_decision: dict[object, list[dict]] = {}
    for o in obs:
        key = o["source_batch_id"] or round(o["decision_stamp"], 6)
        by_decision.setdefault(key, []).append(o)

    out = []
    seen_rounds = set()
    for round_key, group in by_decision.items():
        head = group[0]
        caps = tuple(sorted({round(o["obs_stamp"], 6) for o in group
                             if math.isfinite(o["obs_stamp"])}))
        identity = head["source_batch_id"] or caps
        if dedupe and identity:
            if identity in seen_rounds:
                continue
            seen_rounds.add(identity)
        # The fused answer describes `fused_stamp` where the run records it, and the
        # newest capture time where it does not (schema 1 predates the field).
        when = head["fused_stamp"]
        if not math.isfinite(when) and schema_version(run) < 4:
            when = max(caps) if caps else head["decision_stamp"]
        gx, gy = truth.at([when])
        if not np.isfinite([head["fused_x"], head["fused_y"], gx[0], gy[0]]).all():
            continue
        fused_truth = np.array([float(gx[0]), float(gy[0])])
        cameras = {}
        for o in group:
            cgx, cgy = truth.at([o["obs_stamp"]])
            if not math.isfinite(cgx[0]):
                continue
            cameras[o["camera"]] = dict(
                error_cm=float(np.linalg.norm(
                    np.array([o["obs_x"], o["obs_y"]])
                    - np.array([float(cgx[0]), float(cgy[0])])) * 100.0),
                used=o["used"])
        out.append(dict(
            source_batch_id=head["source_batch_id"],
            decision_stamp=head["decision_stamp"], fused_stamp=when,
            fused_xy=np.array([head["fused_x"], head["fused_y"]]),
            fused_cov=head["fused_cov"], truth=fused_truth,
            error_cm=float(np.linalg.norm(
                np.array([head["fused_x"], head["fused_y"]]) - fused_truth) * 100.0),
            n_candidates=head["n_candidates"], n_used=head["n_used"],
            cameras=cameras))
    return sorted(out, key=lambda r: (r["fused_stamp"], r["source_batch_id"]))


def _integer(value, name):
    if isinstance(value, bool) or str(value).strip() != str(int(value)) or int(value) < 0:
        raise ValueError(f"invalid nonnegative integer {name}")
    return int(value)


def _true(value):
    return value is True or value in ("true", "True", "1", 1)


def belief_at_fusion_events(run: Path, table: list[dict] | None = None, *,
                            max_reference_gap_s=None, reference_frame=None) -> list[dict]:
    """Score explicit committed posteriors; periodic predictions cannot identify them.

    Legacy logs without a v2 posterior record are unavailable for this quantity,
    even when a later public belief happens to have the same or a nearby timestamp.
    Missing truth support retains the identified event with a NaN error and an
    explicit ``reference_supported=False`` flag.
    """
    ledger = validate_run_ledger(run)
    accepted = [a for a in assimilations(run) if a["source_batch_id"] in ledger.accepted_update_ids]
    if not accepted:
        return []
    manifest = json.loads((Path(run) / "run_manifest.json").read_text())
    reference_frame = reference_frame or manifest.get("gt_frame_id")
    required = ("schema_version", "posterior_mean", "posterior_covariance", "state_stamp_ns",
                "epoch", "revision_before", "revision_after", "frame_id", "valid", "motion_supported")
    for a in accepted:
        if any(a.get(k) in (None, "") for k in required) or str(a["schema_version"]) != "2":
            raise PosteriorUnavailable(f"{run}: explicit v2 committed posterior missing for {a['source_batch_id']}")
    if not reference_frame:
        raise PosteriorUnavailable(f"{run}: explicit ground-truth reference frame is required")
    truth = truth_series(run, table, max_reference_gap_s=max_reference_gap_s)
    candidates = {o["source_batch_id"]: o["n_candidates"] for o in observations(run)}
    revisions = set()
    out = []
    for a in accepted:
        mean = np.asarray(json.loads(a["posterior_mean"]), dtype=float)
        P = np.asarray(json.loads(a["posterior_covariance"]), dtype=float)
        if mean.shape != (3,) or not np.isfinite(mean).all():
            raise ValueError("invalid committed posterior mean")
        validate_covariances(P[None], dimension=3)
        stamp_ns = _integer(a["state_stamp_ns"], "state_stamp_ns")
        revision = _integer(a["revision_after"], "revision_after")
        before = _integer(a["revision_before"], "revision_before")
        if revision <= before or (a["epoch"], revision) in revisions:
            raise ValueError("committed posterior revision is not a unique advancing update")
        revisions.add((a["epoch"], revision))
        if a["frame_id"] != reference_frame:
            raise ValueError("posterior/reference frame mismatch")
        if not _true(a["valid"]) or not _true(a["motion_supported"]):
            raise PosteriorUnavailable(f"{run}: committed posterior is invalid or motion-unsupported")
        target = stamp_ns / 1e9
        gx, gy = truth.at([target])
        supported = bool(np.isfinite([gx[0], gy[0]]).all())
        out.append(dict(source_batch_id=a["source_batch_id"], assimilation_status=a["status"],
            n_candidates=candidates.get(a["source_batch_id"], math.nan),
            state_stamp_ns=stamp_ns, planner_belief_stamp=target, frame_id=a["frame_id"],
            epoch=a["epoch"], revision=revision, mean=mean, covariance=P,
            error_cm=float(np.linalg.norm(mean[:2]-[gx[0], gy[0]])*100),
            stated_sigma_cm=float(np.sqrt(np.trace(P[:2, :2])/2)*100),
            belief_lag_after_fusion_s=target-a["correction_stamp"],
            reference_supported=supported, reference_max_gap_s=max_reference_gap_s,
            truth_source=truth.source, quantity="committed_posterior"))
    return sorted(out, key=lambda r: (r["epoch"], r["revision"]))


def validate_covariances(covariances, *, dimension=2):
    """Require full finite, symmetric, positive-definite covariance matrices."""
    c = np.asarray(covariances, dtype=float)
    if c.ndim != 3 or c.shape[1:] != (dimension, dimension):
        raise ValueError(f"expected (n,{dimension},{dimension}) covariance array")
    if not np.isfinite(c).all():
        raise ValueError("nonfinite covariance")
    if not np.allclose(c, np.swapaxes(c, 1, 2), rtol=1e-10, atol=1e-12):
        raise ValueError("asymmetric covariance")
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be positive definite") from exc
    return c


def nees(residuals, covariances) -> np.ndarray:
    """Planar normalized squared error using the complete checked covariance."""
    r = np.asarray(residuals, dtype=float)
    c = np.asarray(covariances, dtype=float)
    if r.size == c.size == 0:
        return np.empty(0)
    if r.ndim != 2 or r.shape[1] != 2 or len(c) != len(r) or not np.isfinite(r).all():
        raise ValueError("expected matching finite planar residuals and covariances")
    validate_covariances(c)
    return np.einsum("ni,ni->n", r, np.linalg.solve(c, r[..., None])[..., 0])


#: Mean of a 2-D chi-square: the target for a MEAN NEES.
NEES_MEAN_TARGET = 2.0
#: Median of a 2-D chi-square: the target for a MEDIAN NEES. Not 2.0 -- comparing a
#: median NEES against 2.0 understates the miscalibration by 44%.
NEES_MEDIAN_TARGET = 2.0 * math.log(2.0)
#: 95th percentile of a 2-D chi-square, for ellipse coverage.
CHI2_95_2D = 5.991
