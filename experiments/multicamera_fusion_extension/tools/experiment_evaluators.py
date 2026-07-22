#!/usr/bin/env python3
"""Offline experiment evaluators for the multi-camera fusion campaign (E1, E2).

This is STUDY code, not runtime: it is deliberately kept out of the
``reliability`` package because the leakage firewall pre-registers
``reliability.evaluation`` as a forbidden planner-facing import
(``src/reliability/config/leakage_firewall.yaml``). Experiment evaluators may
score residuals/predictions that were derived from evaluation ground truth, so
they must never sit on the planner-importable surface. They live here per the
repo convention that study code belongs in ``experiments/<study>/``.

It is the *fixed measurement apparatus* for
``experiments/multicamera_fusion_extension/plans/11_experiment_campaign.md`` and
is data-independent: it scores whatever predictions/residuals a caller supplies,
and is validated against synthetic data with known ground truth so the same
functions run unchanged the moment real fitted models and recorded runs exist.
No fitting and no ground-truth *loading* happen here.

E2 — conditional measurement calibration (§15 E2, plan 06, gate §21):
    Are predicted 2×2 measurement covariances statistically consistent with the
    observed residuals? Reported via mean negative log-likelihood (MNLL), χ² 95%
    coverage, and sharpness (mean log|R|). Coverage and sharpness are reported
    *together* on purpose: a covariance that is enormous everywhere trivially
    achieves high coverage while being useless.

E1 — spatial reliability prediction (§15 E1, plans 03/04/05, gate §21):
    Does a predicted per-observation trust probability match the binary
    usable-observation label? Scored with the canonical scorers in
    ``scripts/shared/metrics.py`` (Brier / NLL / ECE / AUROC / AUPRC /
    false-high-trust-rate). The gate requires the combined GP+confidence model
    to beat both GP-only and confidence-only.

All scoring reuses ``reliability.conditional_covariance`` (canonical MNLL /
coverage / sharpness), ``scripts/shared/metrics.py`` (canonical classification
scorers) and ``reliability.campaign_statistics`` (canonical percentile /
clustered-bootstrap); nothing is re-derived.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "reliability"))

from reliability.conditional_covariance import chi2_coverage, matrix_nll, sharpness  # noqa: E402
from reliability.campaign_statistics import (  # noqa: E402
    Leaf,
    ProportionEstimate,
    Verdict,
    hierarchical_bootstrap,
    mean,
    median,
    percentile,
    wilson_interval,
)

Pair = Sequence[float]
Matrix2 = Sequence[Sequence[float]]

NOMINAL_COVERAGE = 0.95
# A method whose coverage sits at/above this while its sharpness is far worse
# than the sharpest method is over-inflated: high coverage bought with useless
# width (the §E2 failure mode).
_OVERINFLATION_COVERAGE = 0.99


@dataclass(frozen=True)
class CovarianceCalibrationResult:
    """Per-method conditional-covariance calibration summary (E2)."""

    method: str
    n: int
    mnll: float
    coverage_c95: float
    sharpness: float
    bias_x: float
    bias_y: float
    rmse: float
    median_error: float
    p95_error: float
    coverage_target: float = NOMINAL_COVERAGE
    mnll_ci_low: float | None = None
    mnll_ci_high: float | None = None

    @property
    def coverage_gap(self) -> float:
        """Signed distance of empirical coverage from the nominal target."""

        return self.coverage_c95 - self.coverage_target

    def as_row(self) -> dict[str, object]:
        """A reporting row that always carries coverage AND sharpness together."""

        return {
            "method": self.method,
            "n": self.n,
            "mnll": self.mnll,
            "mnll_ci": (self.mnll_ci_low, self.mnll_ci_high),
            "coverage_c95": self.coverage_c95,
            "coverage_gap": self.coverage_gap,
            "sharpness_logdetR": self.sharpness,
            "bias_x": self.bias_x,
            "bias_y": self.bias_y,
            "rmse": self.rmse,
            "median_error": self.median_error,
            "p95_error": self.p95_error,
        }


def _norm(residual: Pair) -> float:
    return math.hypot(float(residual[0]), float(residual[1]))


def evaluate_covariance_calibration(
    method: str,
    residuals: Sequence[Pair],
    covariances: Sequence[Matrix2],
    *,
    groups: Sequence[str] | None = None,
    coverage_q: float = NOMINAL_COVERAGE,
    n_boot: int = 2000,
    seed: int = 0,
) -> CovarianceCalibrationResult:
    """Score one method's predicted covariances against observed residuals (E2).

    ``residuals[t]`` is the 2-vector ``z_t − h(x_t)`` and ``covariances[t]`` is
    the predicted 2×2 ``R_t`` for that sample. When ``groups`` (one run/route key
    per sample) is given, a clustered bootstrap over groups yields a CI on MNLL
    — respecting the run-level unit of analysis (§16) rather than treating frames
    as independent.
    """

    if len(residuals) != len(covariances):
        raise ValueError("residuals and covariances must have equal length")
    if len(residuals) < 2:
        raise ValueError("need at least two samples to evaluate calibration")

    mnll = matrix_nll(residuals, covariances)
    coverage = chi2_coverage(residuals, covariances, coverage_q)
    sharp = sharpness(covariances)

    xs = [float(r[0]) for r in residuals]
    ys = [float(r[1]) for r in residuals]
    norms = [_norm(r) for r in residuals]
    rmse = math.sqrt(mean([n * n for n in norms]))

    mnll_low: float | None = None
    mnll_high: float | None = None
    if groups is not None:
        if len(groups) != len(residuals):
            raise ValueError("groups must align with residuals")
        # Per-sample NLL reuses the canonical formula (matrix_nll on a singleton
        # equals that sample's term), then a clustered bootstrap over groups.
        leaves = [
            Leaf(
                route=str(groups[i]),
                seed="0",
                episode=str(i),
                delta=matrix_nll([residuals[i]], [covariances[i]]),
            )
            for i in range(len(residuals))
        ]
        boot = hierarchical_bootstrap(
            leaves, statistic=mean, n_boot=n_boot, seed=seed, lower_is_better=True
        )
        mnll_low, mnll_high = boot.low, boot.high

    return CovarianceCalibrationResult(
        method=str(method),
        n=len(residuals),
        mnll=mnll,
        coverage_c95=coverage,
        sharpness=sharp,
        bias_x=mean(xs),
        bias_y=mean(ys),
        rmse=rmse,
        median_error=median(norms),
        p95_error=percentile(norms, 0.95),
        coverage_target=coverage_q,
        mnll_ci_low=mnll_low,
        mnll_ci_high=mnll_high,
    )


@dataclass(frozen=True)
class CovarianceComparison:
    """A coverage+sharpness-together table over competing covariance methods."""

    rows: tuple[dict[str, object], ...]
    best_mnll_method: str
    sharpest_method: str
    overinflated_methods: tuple[str, ...]

    def render_markdown(self) -> str:
        header = (
            "| method | n | MNLL | coverage C95 (gap) | sharpness log|R| "
            "| bias x/y | RMSE | p95 |\n"
            "|---|---|---|---|---|---|---|---|"
        )
        lines = [header]
        for row in self.rows:
            ci = row["mnll_ci"]
            ci_txt = (
                f" [{ci[0]:.3f},{ci[1]:.3f}]"
                if ci and ci[0] is not None and ci[1] is not None
                else ""
            )
            lines.append(
                "| {method} | {n} | {mnll:.3f}{ci} | {cov:.3f} ({gap:+.3f}) "
                "| {sharp:.3f} | {bx:+.3f}/{by:+.3f} | {rmse:.3f} | {p95:.3f} |".format(
                    method=row["method"],
                    n=row["n"],
                    mnll=row["mnll"],
                    ci=ci_txt,
                    cov=row["coverage_c95"],
                    gap=row["coverage_gap"],
                    sharp=row["sharpness_logdetR"],
                    bx=row["bias_x"],
                    by=row["bias_y"],
                    rmse=row["rmse"],
                    p95=row["p95_error"],
                )
            )
        return "\n".join(lines)


def compare_covariance_methods(
    results: Sequence[CovarianceCalibrationResult],
    *,
    overinflation_sharpness_margin: float = 1.0,
) -> CovarianceComparison:
    """Package per-method E2 results and flag over-inflated covariances.

    An over-inflated method is one whose coverage sits at/above
    ``_OVERINFLATION_COVERAGE`` yet whose sharpness exceeds the sharpest
    method's by more than ``overinflation_sharpness_margin`` nats — i.e. it
    "passes" coverage only by being uselessly wide (§E2).
    """

    if not results:
        raise ValueError("need at least one method result to compare")
    rows = tuple(r.as_row() for r in results)
    best_mnll = min(results, key=lambda r: r.mnll).method
    sharpest = min(results, key=lambda r: r.sharpness)
    overinflated = tuple(
        r.method
        for r in results
        if r.coverage_c95 >= _OVERINFLATION_COVERAGE
        and (r.sharpness - sharpest.sharpness) > overinflation_sharpness_margin
    )
    return CovarianceComparison(
        rows=rows,
        best_mnll_method=best_mnll,
        sharpest_method=sharpest.method,
        overinflated_methods=overinflated,
    )


# --------------------------------------------------------------------------- #
# E1 — spatial reliability prediction.
# Scored with the CANONICAL scorers in scripts/shared/metrics.py (Brier /
# log-loss=NLL / ECE / AUROC / AUPRC / false-high-trust-rate) — never
# re-derived here, per the repo's anti-fork rule. metrics.py is located
# relative to this file so the tool is runnable from anywhere.
# --------------------------------------------------------------------------- #
DEFAULT_FALSE_TRUST_TAUS = (0.8, 0.9, 0.95)


def _metrics_module():
    try:
        module = importlib.import_module("metrics")
    except ModuleNotFoundError:
        shared = ROOT / "scripts" / "shared"
        if shared.is_dir() and str(shared) not in sys.path:
            sys.path.insert(0, str(shared))
        module = importlib.import_module("metrics")
    if not hasattr(module, "brier") or not hasattr(module, "auprc"):
        raise ImportError(
            "resolved a 'metrics' module without the canonical scorers; "
            "expected scripts/shared/metrics.py"
        )
    return module


@dataclass(frozen=True)
class ReliabilityPredictionResult:
    """Per-method reliability-prediction scores (E1)."""

    method: str
    n: int
    brier: float
    nll: float
    ece: float
    auroc: float
    auprc: float
    false_trust: dict[float, float]
    brier_ci_low: float | None = None
    brier_ci_high: float | None = None

    def as_row(self) -> dict[str, object]:
        return {
            "method": self.method,
            "n": self.n,
            "brier": self.brier,
            "brier_ci": (self.brier_ci_low, self.brier_ci_high),
            "nll": self.nll,
            "ece": self.ece,
            "auroc": self.auroc,
            "auprc": self.auprc,
            "false_trust": dict(self.false_trust),
        }


def evaluate_reliability_prediction(
    method: str,
    probabilities: Sequence[float],
    labels: Sequence[float],
    *,
    groups: Sequence[str] | None = None,
    false_trust_taus: Sequence[float] = DEFAULT_FALSE_TRUST_TAUS,
    n_boot: int = 2000,
    seed: int = 0,
) -> ReliabilityPredictionResult:
    """Score one method's predicted trust probabilities against binary labels.

    ``labels[t] ∈ {0, 1}`` is the usable-observation outcome; ``probabilities[t]``
    is the predicted P(usable). ``groups`` (a run/route key per sample) triggers
    a clustered bootstrap CI on Brier over runs (§16 unit of analysis).
    """

    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have equal length")
    if len(labels) < 2:
        raise ValueError("need at least two samples to evaluate prediction")
    probs = [float(p) for p in probabilities]
    ys = [float(y) for y in labels]
    for p in probs:
        if not 0.0 <= p <= 1.0 or not math.isfinite(p):
            raise ValueError("probabilities must be finite and in [0, 1]")
    for y in ys:
        if y not in (0.0, 1.0):
            raise ValueError("labels must be binary 0/1")

    M = _metrics_module()
    false_trust = {
        float(tau): float(M.fhtr(ys, probs, tau_high=float(tau))) for tau in false_trust_taus
    }

    brier_low: float | None = None
    brier_high: float | None = None
    if groups is not None:
        if len(groups) != len(labels):
            raise ValueError("groups must align with labels")
        leaves = [
            Leaf(route=str(groups[i]), seed="0", episode=str(i), delta=(probs[i] - ys[i]) ** 2)
            for i in range(len(labels))
        ]
        boot = hierarchical_bootstrap(
            leaves, statistic=mean, n_boot=n_boot, seed=seed, lower_is_better=True
        )
        brier_low, brier_high = boot.low, boot.high

    return ReliabilityPredictionResult(
        method=str(method),
        n=len(labels),
        brier=float(M.brier(ys, probs)),
        nll=float(M.logloss(ys, probs)),
        ece=float(M.ece(ys, probs)),
        auroc=float(M.auroc(ys, probs)),
        auprc=float(M.auprc(ys, probs)),
        false_trust=false_trust,
        brier_ci_low=brier_low,
        brier_ci_high=brier_high,
    )


def reliability_prediction_gate(
    combined: ReliabilityPredictionResult,
    gp_only: ReliabilityPredictionResult,
    confidence_only: ReliabilityPredictionResult,
) -> Verdict:
    """E1 acceptance gate (§21 GP gate / E1): the combined GP+confidence model
    must beat BOTH the GP-only and confidence-only models on grouped test data,
    on Brier AND NLL. Otherwise the combination adds complexity without evidence.
    """

    criteria = {
        "beats_gp_only_brier": combined.brier < gp_only.brier,
        "beats_confidence_only_brier": combined.brier < confidence_only.brier,
        "beats_gp_only_nll": combined.nll < gp_only.nll,
        "beats_confidence_only_nll": combined.nll < confidence_only.nll,
    }
    return Verdict(
        passed=all(criteria.values()),
        criteria=criteria,
        notes="§21 E1 combined-beats-both gate",
    )


# --------------------------------------------------------------------------- #
# E5 / E6 — fault detection & isolation from the health-state timeline.
#
# The health monitor (reliability.health_ewma, run inside
# ReplayMode.HEALTH_AWARE_FUSION) uses NO ground truth. THIS scoring does: it
# compares the monitor's DEGRADED flags against the KNOWN injected fault (a
# controlled experimental variable, not sensor GT) — which is exactly why it is
# firewalled study code. It feeds the pre-registered detection gate and the
# critical-failure stop rule (a monitor that removes healthy cameras at least as
# often as it catches faulty ones must not be used downstream).
# --------------------------------------------------------------------------- #

_DEGRADED = "DEGRADED"
_NON_HEALTHY = ("SUSPECT", "DEGRADED", "RECOVERING")
# ≥3 co-visible cameras are needed to attribute a fault to ONE camera; with 2 the
# evidence is ambiguous (pre-reg §2 / §11.3 caveat), so isolation is not scored.
_MIN_CAMERAS_FOR_ISOLATION = 3
DEFAULT_NOMINAL_FAR_GATE = 0.02  # pre-reg §5 (WP5 measured 0.000 over 3700 frames)

# One health-state timeline entry: (timestamp_s, {camera_id: state}). A caller
# builds it from a replay result via
# ``[(s.timestamp_s, s.health_state_by_camera) for s in result.steps]``.
HealthTimeline = Sequence["tuple[float, Mapping[str, str]]"]


@dataclass(frozen=True)
class FaultDetectionResult:
    """Per-episode fault-detection outcome (E5/E6)."""

    episode: str
    faulted_camera: str | None
    onset_s: float | None
    n_cameras: int
    detected: bool
    detection_delay_s: float | None
    escalation_delay_s: float | None  # onset -> first non-HEALTHY of the faulted camera
    isolated: bool | None  # None when nominal or < 3 cameras (ambiguous)
    false_alarm: bool
    false_alarm_cameras: tuple[str, ...]

    def as_row(self) -> dict[str, object]:
        return {
            "episode": self.episode,
            "faulted_camera": self.faulted_camera,
            "onset_s": self.onset_s,
            "n_cameras": self.n_cameras,
            "detected": self.detected,
            "detection_delay_s": self.detection_delay_s,
            "escalation_delay_s": self.escalation_delay_s,
            "isolated": self.isolated,
            "false_alarm": self.false_alarm,
            "false_alarm_cameras": list(self.false_alarm_cameras),
        }


def _first_state_time(
    steps: Sequence["tuple[float, Mapping[str, str]]"],
    camera: str,
    states: tuple[str, ...],
    *,
    at_or_after: float | None,
) -> float | None:
    for t, by_cam in steps:
        if at_or_after is not None and t < at_or_after:
            continue
        if str(by_cam.get(camera, "")) in states:
            return t
    return None


def evaluate_fault_detection(
    episode: str,
    timeline: HealthTimeline,
    *,
    faulted_camera: str | None = None,
    onset_s: float | None = None,
    degraded_state: str = _DEGRADED,
    min_cameras_for_isolation: int = _MIN_CAMERAS_FOR_ISOLATION,
) -> FaultDetectionResult:
    """Score one episode's health-state timeline against the injected fault.

    ``timeline`` is a sequence of ``(timestamp_s, {camera_id: state})``; states are
    the ``CalibrationHealthState`` values (HEALTHY/SUSPECT/DEGRADED/RECOVERING) that
    ``ReplayStep.health_state_by_camera`` logs. For a **fault** episode pass the
    ``faulted_camera`` and its ``onset_s``; for a **nominal** episode leave both
    ``None`` (only the false-alarm fields are then meaningful).

    Detection is the faulted camera reaching ``degraded_state`` at or after onset;
    ``escalation_delay_s`` is its first non-HEALTHY state (the earlier *inflate*
    signal). ``false_alarm_cameras`` are the *other* cameras that ever went
    DEGRADED — a healthy camera wrongly removed. Isolation (faulted degraded AND no
    other camera degraded) is scored only with ≥ ``min_cameras_for_isolation``
    cameras; otherwise it is ``None`` (ambiguous).
    """

    steps = sorted(
        ((float(t), dict(by_cam)) for t, by_cam in timeline),
        key=lambda item: item[0],
    )
    cameras: set[str] = set()
    for _t, by_cam in steps:
        cameras.update(str(c) for c in by_cam.keys())
    n_cameras = len(cameras)

    if faulted_camera is not None and onset_s is None:
        raise ValueError("a fault episode requires onset_s")

    other_degraded = tuple(
        sorted(
            cam
            for cam in cameras
            if cam != faulted_camera
            and _first_state_time(steps, cam, (degraded_state,), at_or_after=None) is not None
        )
    )
    false_alarm = len(other_degraded) > 0

    if faulted_camera is None:
        # Nominal episode: only false alarms are defined.
        return FaultDetectionResult(
            episode=str(episode),
            faulted_camera=None,
            onset_s=None,
            n_cameras=n_cameras,
            detected=False,
            detection_delay_s=None,
            escalation_delay_s=None,
            isolated=None,
            false_alarm=false_alarm,
            false_alarm_cameras=other_degraded,
        )

    onset = float(onset_s)
    t_degraded = _first_state_time(steps, faulted_camera, (degraded_state,), at_or_after=onset)
    t_escalated = _first_state_time(steps, faulted_camera, _NON_HEALTHY, at_or_after=onset)
    detected = t_degraded is not None
    detection_delay = (t_degraded - onset) if t_degraded is not None else None
    escalation_delay = (t_escalated - onset) if t_escalated is not None else None
    isolated: bool | None
    if n_cameras < int(min_cameras_for_isolation):
        isolated = None
    else:
        isolated = detected and not false_alarm

    return FaultDetectionResult(
        episode=str(episode),
        faulted_camera=str(faulted_camera),
        onset_s=onset,
        n_cameras=n_cameras,
        detected=detected,
        detection_delay_s=detection_delay,
        escalation_delay_s=escalation_delay,
        isolated=isolated,
        false_alarm=false_alarm,
        false_alarm_cameras=other_degraded,
    )


@dataclass(frozen=True)
class FaultDetectionSummary:
    """Campaign-level detection summary + the §5 critical-failure stop rule."""

    n_fault_episodes: int
    n_nominal_episodes: int
    detection_rate: ProportionEstimate | None
    isolation_tpr: ProportionEstimate | None
    false_isolation_rate: ProportionEstimate | None
    nominal_far: ProportionEstimate | None
    median_detection_delay_s: float | None
    mean_detection_delay_s: float | None
    stop_rule: Verdict

    def as_row(self) -> dict[str, object]:
        def _pt(est: ProportionEstimate | None) -> object:
            return None if est is None else {"point": est.point, "low": est.low, "high": est.high}

        return {
            "n_fault_episodes": self.n_fault_episodes,
            "n_nominal_episodes": self.n_nominal_episodes,
            "detection_rate": _pt(self.detection_rate),
            "isolation_tpr": _pt(self.isolation_tpr),
            "false_isolation_rate": _pt(self.false_isolation_rate),
            "nominal_far": _pt(self.nominal_far),
            "median_detection_delay_s": self.median_detection_delay_s,
            "mean_detection_delay_s": self.mean_detection_delay_s,
            "stop_rule_passed": self.stop_rule.passed,
        }


def summarize_fault_detection(
    results: Sequence[FaultDetectionResult],
    *,
    nominal_far_gate: float = DEFAULT_NOMINAL_FAR_GATE,
) -> FaultDetectionSummary:
    """Aggregate per-episode results and evaluate the critical-failure stop rule.

    Stop rule (pre-reg §5, non-negotiable): the health monitor may be used
    downstream only if isolation-TPR strictly exceeds the false-isolation rate
    (it must catch faulty cameras more often than it removes healthy ones) AND the
    nominal false-alarm rate is within ``nominal_far_gate``. Rates use Wilson
    intervals (§16) because the run counts are small.
    """

    fault = [r for r in results if r.faulted_camera is not None]
    nominal = [r for r in results if r.faulted_camera is None]

    detection_rate = (
        wilson_interval(sum(1 for r in fault if r.detected), len(fault)) if fault else None
    )
    iso_eps = [r for r in fault if r.isolated is not None]
    isolation_tpr = (
        wilson_interval(sum(1 for r in iso_eps if r.isolated), len(iso_eps)) if iso_eps else None
    )
    false_isolation_rate = (
        wilson_interval(sum(1 for r in fault if r.false_alarm), len(fault)) if fault else None
    )
    nominal_far = (
        wilson_interval(sum(1 for r in nominal if r.false_alarm), len(nominal)) if nominal else None
    )

    delays = [r.detection_delay_s for r in fault if r.detection_delay_s is not None]
    median_delay = median(delays) if delays else None
    mean_delay = mean(delays) if delays else None

    criteria: dict[str, bool] = {}
    if isolation_tpr is not None and false_isolation_rate is not None:
        criteria["isolation_exceeds_false_isolation"] = (
            isolation_tpr.point > false_isolation_rate.point
        )
    if nominal_far is not None:
        criteria["nominal_far_within_gate"] = nominal_far.point <= float(nominal_far_gate)
    stop_rule = Verdict(
        passed=bool(criteria) and all(criteria.values()),
        criteria=criteria,
        notes=(
            "§5 critical-failure stop rule: isolation-TPR must exceed false-isolation "
            f"rate AND nominal FAR ≤ {float(nominal_far_gate):g}"
        ),
    )

    return FaultDetectionSummary(
        n_fault_episodes=len(fault),
        n_nominal_episodes=len(nominal),
        detection_rate=detection_rate,
        isolation_tpr=isolation_tpr,
        false_isolation_rate=false_isolation_rate,
        nominal_far=nominal_far,
        median_detection_delay_s=median_delay,
        mean_detection_delay_s=mean_delay,
        stop_rule=stop_rule,
    )
