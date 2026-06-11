# Uncertainty propagation in the external-camera EFE navigation system

Canonical, paper-facing description of how odometry, encoder, heading and
observation uncertainty are propagated, and how that couples to the obstacle cost
and the global plan. Written 2026-06-11 to fix recurring terminology confusion and
to document the now-active analytical process-noise model. Supersedes scattered
notes in `runtime_dataflow.md` / `perception_details.md` on these points.

## TL;DR

- There are **three different noise objects**, and they must not be conflated:
  (A) the **planner/filter model** process noise (`process_noise_xy`, `process_noise_theta`),
  (B) the **simulation ground-truth corruptions** (`command_noise_*`, `encoder_noise_*`),
  (C) the **measurement** noise (`r_visible_uv`, `r_miss_uv`, GP-blended).
- Heading is **odometry-driven dead reckoning** in the predict step. The camera
  contributes ground-plane `(x, y)` only; under `camera_xy_only` **no explicit yaw
  measurement is fused**. Heading is corrected only *indirectly*, through the
  position↔heading cross-covariance built by the motion Jacobian and the process noise.
- The planner now uses the **exact integrated process-noise covariance** `Q_d(θ,v,Δt)`
  (the appendix `app:heading` derivation), not a frozen diagonal.
- The GP changes the camera `(x, y)` covariance **only** — never heading.

---

## 1. Three distinct noise families

| family | config keys (`aws_f31b1_final_config.yaml`) | value | where it lives | what it is |
|---|---|---|---|---|
| **A. Model / process noise** (filter + planner) | `process_noise_xy`, `process_noise_theta` | 0.01, 0.046 | `dynamics.py` / `casadi_efe.py` → `Q_d` | the **assumed** actuation-noise spectral densities the belief uses to grow covariance during prediction |
| **B. Simulation corruptions** (ground truth) | `command_noise_*`, `encoder_noise_*` | see config L167–177 | `src/sim/sim/actuation_noise_node.py` (`/cmd_vel`), `src/sim/sim/encoder_noise_node.py` (`/odom_noisy`) | the **real** disturbances injected into the simulated robot; the planner does **not** know them |
| **C. Measurement noise** | `r_visible_uv`, `r_miss_uv` | 2.5, 40.0 (px) | `casadi_efe._blend_observation_covariance_ca` | image-space camera `(x,y)` measurement covariance, blended by GP visibility |

The single most common mistake is treating A and B as the same quantity. They are
not: **B is the world; A is the model of the world.** A good filter has A
*commensurate with* B, but they are different objects with different roles, and they
are deliberately **not** set equal.

---

## 2. The process-noise covariance `Q_d` (now active)

Continuous unicycle with zero-mean Gaussian disturbances on the **actuation
signals** (forward speed `v` and yaw rate `ω`), with power spectral densities
`σ_v² = process_noise_xy²` and `σ_ω² = process_noise_theta²`:

```
ẋ = (v+w_v)cosθ ,  ẏ = (v+w_v)sinθ ,  θ̇ = ω + w_ω
```

The motion Jacobian `F = ∂f/∂s` is **nilpotent** (`F² = 0`), so the matrix
exponential truncates exactly to `e^{FΔt} = I + FΔt`, and the discrete process-noise
covariance integrates in closed form to

```
Q_d = ∫₀^{Δt} e^{Fτ} Q_c e^{Fᵀτ} dτ
    = Q_c Δt + ½(F Q_c + Q_c Fᵀ)Δt² + ⅓ F Q_c Fᵀ Δt³
```

with `Q_c = L(θ) diag(σ_v², σ_ω²) L(θ)ᵀ`. The full matrix (with `c=cosθ`, `s=sinθ`)
is implemented identically in NumPy (`unicycle_process_noise`,
`src/planning/planning/core/dynamics.py:29`) and CasADi
(`unicycle_process_noise_ca`, `src/planning/planning/core/casadi_efe.py:101`), and
the two are asserted equal in `tests/visibility_comparison/test_dynamics_jacobian.py`.

It enters the belief propagation as `S_{k+1} = F S_k Fᵀ + Q_d` — in the EKF predict
(`base_planner.predict`, `base_planner.py:406`) and **per step** inside the EFE
horizon (`casadi_efe.py:374`, rebuilt each step from the predicted `θ` and the
candidate `v`; there is no longer any frozen `Q` field on `CasadiEfeParams`).

**Why this matters (and what changed).** The off-diagonal terms
`q02 = -½ v sinθ σ_ω² Δt²`, `q12 = ½ v cosθ σ_ω² Δt²` are the mechanism by which
**yaw-rate uncertainty leaks into position uncertainty over a step** — they create
the position↔heading cross-covariance that later lets a camera `(x,y)` update also
sharpen heading. The retired implementation used a frozen diagonal
`diag(σ_v², σ_v², σ_ω²)` (treating the parameters as *per-step* std, no `Δt`, no
cross-terms). Switching to `Q_d` reinterprets the parameters as **PSDs (std per √s)**
integrated over `Δt`, so both the magnitude and the shape of the propagated
covariance changed. Offline `F31_b1` EFE totals dropped accordingly (C1 d=0.06/
total≈8001, C2 d=0.34/total≈9886 vs the prior diagonal baseline C1 0.09/11179,
C2 0.09/11823); **all single-version results predating this change are superseded.**

---

## 3. Heading, odometry, encoder — what actually happens

1. **Heading is dead-reckoned from odometry.** With `use_odom_for_predict: true`
   the predict step integrates the (noisy) odometry stream, so the belief heading
   `θ` advances by the measured yaw rate. This is the **architecture**, not a
   "fallback".
2. **No explicit yaw measurement is fused.** Under `heading_update_mode: camera_xy_only`
   the planner's yaw-measurement selector returns `None`
   (`unicycle_planner_node.py:1245`) — the camera update touches `(x, y)` only.
3. **Heading is corrected only indirectly.** Because `F` (and `Q_d`) build the
   cross-covariance `S_xθ, S_yθ`, a camera `(x,y)` correction propagates through the
   Kalman gain into the heading estimate. There is no direct heading observation.
4. **Encoder noise is the real heading-drift source.** `encoder_noise_node.py`
   corrupts `/odom_noisy` (the stream the predict integrates), so heading genuinely
   drifts in closed loop; `actuation_noise_node.py` similarly perturbs executed
   commands. `process_noise_theta` is the belief's **model** of that drift, not the
   drift itself.

**Why this is realistic.** A single ground-projected point from one fixed camera
cannot observe orientation — heading is unobservable from instantaneous position
alone. Real external-camera deployments therefore lean on odometry for heading and
accept that it drifts; the genuine failure mode is heading/position drift in
camera-poor regions, which is exactly what the experiment is designed to expose.

---

## 4. GP scope

The GP maps learned observation reliability to the camera `(x, y)` **measurement**
covariance only (`_blend_observation_covariance_ca`, `casadi_efe.py`). It never
appears in the heading channel and never in the process noise. "Visibility-aware"
means *the camera (x,y) covariance is state-dependent*, nothing more.

---

## 5. Terms used wrong (corrected here)

- **"process_noise_xy / process_noise_theta = position/heading noise"** → they are
  the **actuation** PSDs `σ_v` (forward-speed) and `σ_ω` (yaw-rate). `process_noise_xy`
  is a misleading name; it is *not* an xy-position noise. (Kept as-is in config for
  stability, but read it as `σ_v`.)
- **"σ_θ = 0.046 matches the encoder noise specification"** → false equivalence.
  `process_noise_theta = 0.046` is a **modeled** PSD (family A); the encoder angular
  slip std is `0.16` (family B). They are different objects; one models the other,
  they are not equal. (Flagged for a later TeX fix in §8.)
- **"odometry fallback" for heading** → heading-from-odometry is the **primary**
  architecture under `camera_xy_only`, not a degraded fallback.
- **"the camera measures pose / heading"** → the camera measures image-space `(u,v)`
  → ground `(x,y)` only; heading is odometry + indirect cross-covariance.
- **"process noise ≈ command/encoder noise"** → model (A) vs ground truth (B); never
  conflate. The planner is deliberately blind to B.

---

## 6. Relation to the obstacle (keep-in) cost

The no-go cost is a **keep-in** penalty: the robot is penalised for leaving the
known driveable union. Clearance is now the signed distance to the **true union
boundary** (`occlusion_geometry._get_union_boundary_segments` +
`signed_distance_to_union_xy(..., keep_in=True)`), used by
`nogo_cost.NogoZoneCostModel`. This replaces the old per-prism `min` distance, which
falsely penalised interior **seam** points where two driveable rectangles meet.
Verified on the real `F31_b1` driveable geometry: aisle interiors and the
A3↔A4 connector seam read as deep-inside (clearance ≈0.4 m, no phantom zero), rack
bodies and out-of-lane points read as outside.

Coupling to covariance: the model also supports a **belief-inflated** clearance
`clearance − κ·σ_max(S_xy)` (`penalty_belief_np` / `nogo_belief_kappa`), which is
where the propagated `Q_d`-driven covariance *would* tighten clearance in uncertain
regions. **In the locked paper config this is OFF** (`use_belief_nogo_cost: false`),
so today the global solve uses **mean-only** clearance; `Q_d` affects the obstacle
term only if belief-nogo is later enabled.

---

## 7. Relation to global planning

The global EFE solve (`global_horizon: 120`) propagates the belief `S` forward with
`S_{k+1}=F S Fᵀ + Q_d` at every horizon step. The **risk** and **ambiguity** terms
are functions of that predicted covariance and of the GP visibility evaluated at the
predicted mean (expected over the `(x,y)` belief). Therefore:

- `Q_d` **shapes the route-choice landscape**: more realistic covariance growth in
  camera-poor stretches raises predicted ambiguity/risk there, which is the lever the
  visibility-aware condition (C2) uses to prefer observable behaviour.
- **Obstacle clearance does not depend on `Q_d`** under the locked config (mean-only
  keep-in, §6). Route choice and obstacle feasibility are thus decoupled today: the
  covariance influences *where the plan wants to go* (risk/ambiguity), while the
  keep-in term enforces *staying on driveable floor* using the mean.

---

## 8. Alignment with the paper + stale claims to fix (no TeX edited here)

- **Appendix `app:heading`** ("Discretization of the nonlinear dynamics") matches the
  implemented `Q_d` exactly (nilpotent `F`, 3-term closed form). ✔
- **Appendix `app:state_estimation`**: "the heading column of `G_k` is zero" matches
  `camera_xy_only` (no heading measurement). ✔
- **`PLANNER_HYPERPARAMETERS.md`** lists `process_noise_theta = 0.02`; the active
  config is **0.046**. Reconcile the doc to 0.046 and describe the parameters as PSDs
  (std/√s) integrated over `Δt`.
- **Experimental-setup section** states σ_θ "matches the encoder noise specification".
  Reword: σ_θ is a *modeled* process PSD (family A), commensurate with — not equal to —
  the simulation's encoder angular-slip std 0.16 (family B). (TeX fix deferred.)
