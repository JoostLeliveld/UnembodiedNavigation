# P1-WP5 — self-monitoring single-camera localization

**Claim.** The system detects when the camera no longer provides its expected
service (outage, freeze, drift, delay, degradation) and reacts safely, using NO
ground truth online. This is the checkpoint-C3 module and — under Framing S — a
Paper-1 headline contribution in its own right.

**Serves:** new (no existing chapter owns single-cam health); supports ch.05/06.

## Single-camera caveat (state it up front)
With one camera there is **no cross-camera disagreement** signal — isolation of a
*which-camera* fault is impossible; only self-consistency, stream and bias health
apply. (Two cameras make isolation ambiguous; three make it possible — that is
Paper 2's P2-WP5.) WP5's claim is therefore *degradation detection + safe
degradation*, not fault identification.

## Health evidence
- **Stream:** no frame / frozen sequence / frame age / frame rate.
- **Detection:** miss rate, association-failure rate, box-size drift.
- **Filter consistency:** innovation `ν = z − h(x⁻)`; `S = HP⁻Hᵀ + R`; NIS
  `d² = νᵀS⁻¹ν` (existing 2-dof 0.99 threshold 9.21).
- **Bias:** EWMA innovation `b_t = (1−ρ_b)b_{t−1} + ρ_b ν_t` — the key
  calibration-drift detector.

## Health state machine
`HEALTHY → SUSPECT → DEGRADED → OFFLINE → RECOVERING → HEALTHY`. Actions:
SUSPECT = inflate covariance + more logging; DEGRADED = reject measurements, enter
bounded dead-reckoning, notify planner; OFFLINE = camera unavailable, bounded
continuation or stop; RECOVERING = require consecutive consistent obs, ramp
covariance back down.

## Fault experiments (all CONTROLLED ABLATION)
Full outage; frozen frame (repeat image, current stamps); burst dropout
{0.5,1,2,5 s}; delay {0,100,300,500,1000 ms}; calibration perturbation
(yaw/pitch/translation/homography noise — software ablation, images unchanged);
blur/occlusion; and one **nominal long run** to measure false alarms.

## Metrics
Fault-detection delay; max localization error before detection;
dangerous-overconfidence duration; false-alarm rate; recovery time; state-transition
accuracy; dead-reckoning survival time; % time nominal/degraded/unavailable.

## Gate G5 (= checkpoint C3)
Severe faults increase uncertainty or trigger rejection; false alarms below
threshold; recovery stable; planner receives an explicit localization status; the
health logic uses NO GT. **Critical failure criterion (stop rule 6):** if the
monitor degrades the healthy camera more than it catches real faults, it is not
used downstream. C3-no fallback: limit Paper 1 to nominal service; move health
monitoring fully into Paper 2.

## Reuse (no new algorithm)
`reliability.health_ewma` — `InnovationHealthMonitor` (NIS/bias EWMA, drop-rate
window), `HealthDebouncer` (`CalibrationHealthState`, `response_policy`). Built for
multicam; the single-cam monitor is the subset with the cross-camera term set to
zero. Implementation note (when built): add `OFFLINE` to the enum or map it onto a
stream-dead flag. Fault injection = a study tool; do not hand-roll a new filter.
Outputs → `logs/studies/single_camera_uigp_reliability/wp5_self_monitoring/RESULTS.md`.
