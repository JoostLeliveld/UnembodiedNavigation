"""Reference-anchored localization cost -- the corrected observability term.

The problem this replaces
------------------------
The EFE ambiguity term is the expected conditional entropy of the predicted
observation given the state,

    A_t = 0.5 * ( d*log(2*pi*e) + logdet( Sigma_y - Gamma^T S^-1 Gamma ) ).

Under the ET1 (first-order extended transform) used by this planner the
conditional covariance collapses *exactly* to the planner-facing measurement
covariance:

    Sigma_y = J S J^T + R_plan,   Gamma = S J^T
    => Sigma_y - Gamma^T S^-1 Gamma = J S J^T + R_plan - J S J^T = R_plan

so, exactly,

    A_t = 0.5*d*log(2*pi*e) + 0.5*logdet(R_plan(q_t, R_t)).                (1)

Two consequences follow, and both are accounting bugs rather than modelling
choices:

1. `A_t` carries a large route-independent additive constant,
   `A_ref = 0.5*d*log(2*pi*e) + 0.5*logdet(R_ref)`. In a *fixed-horizon*
   comparison (IWAI) every candidate accumulates exactly the same number of
   copies of that constant, so it cancels and is invisible. In a *full-route*
   comparison, candidates have different arrival times T, so the constant
   contributes `A_ref * T` -- a pure duration term with an arbitrary sign. If
   `A_ref < 0` (which happens whenever `det(R_ref) < (2*pi*e)^-d`, i.e. purely as
   a function of the units the measurement is expressed in) a longer route is
   *rewarded* for taking longer. The route ranking then depends on whether the
   camera measurement is written in pixels or in normalised image coordinates,
   which is indefensible.

2. Even where the constant happens to be positive, it acts as an accidental,
   unit-dependent travel cost. The travel baseline must be explicit.

The correction
--------------
Keep exactly the q/R-dependent part of (1) and drop the constant, by measuring
the ambiguity *relative to a fixed reference measurement quality* `R_ref`:

    c_loc(t) = max( A_t - A_ref, 0 )
             = max( 0.5 * log( det(R_plan(q_t, R_t)) / det(R_ref) ), 0 )    (2)

Properties (each is covered by a regression test):

* `c_loc = 0` exactly when q = 1 and R = R_ref: constant, perfect observability
  contributes no route-dependent preference at all, so route choice is decided
  by the explicit travel baseline -- the shortest safe route wins.
* `c_loc >= 0` always: poor observability can only ever *add* cost. No route can
  be rewarded for taking longer, whatever q and R do.
* `c_loc` is strictly increasing in `det(R_plan)`, so lowering q (availability)
  or worsening R (quality when available) strictly increases it.
* `c_loc` is invariant to any constant offset added to the ambiguity, because the
  same offset appears in `A_t` and in `A_ref` and cancels.
* `R_ref` is the *best attainable* measurement covariance, so the clamp in (2) is
  inactive in normal use; it is there to make the no-reward property structural
  rather than configuration-dependent.

Units: nats.

Where each form is used:

* The **fixed-horizon MPC** (`base_planner._evaluate_controls` and its CasADi
  twin) uses `c_loc` per step, exactly like the risk term, because that objective
  is a per-step discounted average. At a fixed horizon the anchored term differs
  from the raw ambiguity by a route-independent constant, so the optimum and the
  gradients are unchanged -- the correction cannot destabilise the controller, it
  only removes a constant that has no business being there.
* The **full-route selector** (`global_route_selector`) integrates `c_loc` over
  time (nat-seconds) and converts it to seconds-equivalent with one explicit
  weight, because there the candidates have different durations and the objective
  is expressed in absolute time.
"""

from __future__ import annotations

import math

import numpy as np


def ambiguity_constant(dim: int) -> float:
    """The route-independent additive constant of the EFE ambiguity term."""
    return 0.5 * float(dim) * math.log(2.0 * math.pi * math.e)


def reference_ambiguity(R_ref) -> float:
    """`A_ref`: the ambiguity a step would have at the reference measurement quality."""
    R_ref = np.asarray(R_ref, dtype=float)
    sign, logdet = np.linalg.slogdet(R_ref)
    if sign <= 0:
        logdet = math.log(max(float(np.linalg.det(R_ref)), 1e-12))
    return ambiguity_constant(R_ref.shape[0]) + 0.5 * float(logdet)


def localization_cost_rate_np(R_plan, R_ref, *, clamp_nonnegative: bool = True) -> float:
    """Corrected per-step localization cost `c_loc` in nats (equation (2))."""
    R_plan = np.asarray(R_plan, dtype=float)
    R_ref = np.asarray(R_ref, dtype=float)
    if R_plan.shape != R_ref.shape:
        raise ValueError(
            f"R_plan {R_plan.shape} and R_ref {R_ref.shape} must have the same shape"
        )
    sign_p, logdet_p = np.linalg.slogdet(R_plan)
    if sign_p <= 0:
        logdet_p = math.log(max(float(np.linalg.det(R_plan)), 1e-12))
    sign_r, logdet_r = np.linalg.slogdet(R_ref)
    if sign_r <= 0:
        logdet_r = math.log(max(float(np.linalg.det(R_ref)), 1e-12))
    rate = 0.5 * (float(logdet_p) - float(logdet_r))
    if clamp_nonnegative:
        return float(max(rate, 0.0))
    return float(rate)


def localization_cost_rate_from_ambiguity(ambiguity_value: float, R_ref, *, clamp_nonnegative: bool = True) -> float:
    """Same quantity, obtained by anchoring an already-computed ambiguity value.

    Shows the correction in its simplest form: subtract the reference ambiguity.
    Used by the NumPy evaluator so the corrected term is provably the anchored
    version of the very ambiguity value the legacy objective used.
    """
    rate = float(ambiguity_value) - reference_ambiguity(R_ref)
    if clamp_nonnegative:
        return float(max(rate, 0.0))
    return float(rate)


def make_localization_cost_rate_ca(R_ref, *, clamp_nonnegative: bool = True):
    """CasADi twin of `localization_cost_rate_np` for a 2x2 measurement covariance.

    Returns a callable mapping a symbolic 2x2 `R_plan` to the scalar rate, so the
    NumPy and CasADi objectives share one definition (and one regression test
    asserts they agree numerically).
    """
    import casadi as ca

    R_ref = np.asarray(R_ref, dtype=float)
    if R_ref.shape != (2, 2):
        raise ValueError("CasADi localization cost currently supports 2x2 measurement covariances")
    logdet_ref = float(np.log(max(float(np.linalg.det(R_ref)), 1e-12)))

    def rate(R_plan):
        det_plan = R_plan[0, 0] * R_plan[1, 1] - R_plan[0, 1] * R_plan[1, 0]
        value = 0.5 * (ca.log(ca.fmax(det_plan, 1e-12)) - logdet_ref)
        if clamp_nonnegative:
            return ca.fmax(value, 0.0)
        return value

    return rate
