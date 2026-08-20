# Methodology

Companion to `00_problem_statement.md`. Covers the parts that are method-of-record:
what the network exposes, how it is fused, how a planner consumes it, how the network itself
is varied, and how each is scored. Estimation of the fields is an **input** here, not the
contribution — it is specified only as far as the planning and fusion questions require.

Sections marked **[REQUIRED, NOT YET RUN]** state a protocol whose preconditions are not
currently met; nothing may be claimed from them until they are.

---

## M1. What the network exposes

For camera `i` and a candidate position `p`, three quantities are kept distinct:

| Symbol | Meaning | Consumed by |
|---|---|---|
| `V_i ∈ {0,1}` | a clear sight-line exists | nothing directly; it is latent |
| `D_i ∈ {0,1}` | a **usable detection** arrives | the planner, through availability |
| `y_i ∼ N(h_i(p), R_i)` given `D_i=1` | the measurement | the filter |

The planning-relevant field is availability,
`p_use,i(p | map) = Pr(D_i = 1 | p, map)`, which factorises into a term that moves when the
building moves and one that moves when the light changes:

```
p_use,i  =  Pr(V_i = 1 | p, map)  ·  Pr(D_i = 1 | V_i = 1, appearance)
              └─ geometry ─┘            └────── imaging ──────┘
```

The second factor is empirically flat in this environment (problem statement §6.1), so the
imaging term is held constant and reported as a measured null rather than modelled.

**A missing detection contributes no update.** It is never represented as a measurement with
inflated covariance — that requires an arbitrary miss endpoint and conflates availability with
conditional accuracy, which are demonstrably different fields (§3a). `R_i` is held at its
commissioned value throughout and no claim is made about it.

## M2. Fusion

Fusion is treated as a first-class part of the method, because it determines both what the
planner reads and what the filter can know.

### M2.1 Availability fuses by noisy-OR

Across a camera subset `S`, the probability that *some* camera delivers an update is

```
p_any(p) = 1 − Π_{i∈S} (1 − p̂_i(p))
```

Each camera's raw score is mapped to a probability by a two-parameter logistic link fitted
once on a held-out calibration partition and then **frozen** — a deployed system does not get
to recalibrate against the outcomes it is being scored on.

**The consequence that drives method choice.** Noisy-OR is a product of complements, so it
amplifies any per-camera failure to assert "definitely not visible". A field with per-camera
floor `f` has fused floor `1 − (1−f)^|S|`. This is why estimator selection **must not** be
done on per-camera average-case score: monocular depth ties the surveyed ray-cast on held-out
Brier yet its floor rises 0.057 → 0.208 over four cameras, erasing the blackspots a planner
exists to avoid. The link function is not the culprit — isotonic regression moves the fused
floor only to 0.183. Therefore:

> **Rule.** Any estimator comparison intended to inform planning reports the **fused floor at
> the deployed camera count**, alongside any aggregate score. An estimator that ties on Brier
> and loses its fused tail is reported as *not usable for avoidance at that density*, not as
> a tie.

### M2.2 Position fuses recursively, and the transport is a design variable

Two transports exist in this stack and they are **not equivalent**:

| Transport | What arrives | Heading observability |
|---|---|---|
| Per-camera image-space (`pixel_pose`) | a measurement in image coordinates | **observed** — the projection Jacobian has non-zero heading rows |
| Pre-fused world-frame (`/state/bev`) | a metric x,y position | **none** — an xy fix says nothing about heading |

The second is what the multi-camera path uses, and it stamps a non-informative yaw variance
on every correction. That is a correct declaration, not a bug; the defect is that nothing
replaces the lost observability, and a belief-space planner that inflates obstacle clearance
by the predicted covariance is extremely sensitive to exactly that term (§3d).

> **Rule.** A fusion architecture is reported together with the state variables it leaves
> observable. Any closed-loop result must record the belief covariance actually carried at
> plan time, per state variable, and a run whose heading variance sits at its non-informative
> prior is a **failed precondition**, not a data point.

### M2.3 Network design is an experimental axis

The camera subset is varied, not fixed: `{4, 3, 2-opposite, 2-same-wall, 1}`, declared before
any outcome was seen. Count and geometry are deliberately separated — the two 2-camera
subsets hold count constant and vary complementarity. This is what exposes that the benefit
of availability modelling tracks *placement*, not count (§3c).

## M3. How a planner consumes the field

### M3.1 The decision rule

A better field is worth nothing unless a rule acts on it. Let `route` be a path over the lane
network, `len(route)` its length, and

```
blind(route | p̂) = ∫_route (1 − p̂_any) dℓ
```

its **expected blind distance** — the metres it expects to drive with no camera on it. The
planner solves

```
route*(ε) = argmin_route  blind(route | p̂)     s.t.  len(route) ≤ (1+ε)·len(route_min)
```

with `ε` a declared detour budget. One parameter, identical for every arm, degenerating to
the shortest route at `ε = 0`. Solved by sweeping a Lagrange weight `λ` in an edge cost
`1 + λ(1 − p̂_any)` on the 0.25 m lane graph and keeping the feasible route of least predicted
blind distance.

The planner reads **only its own estimator's** `p̂`. The returned route is then scored against
**real detector outcomes**, which no planner sees.

### M3.2 Candidate routes must be a real menu

A route comparison is only informative if the arms could have disagreed. The candidate set is
therefore enumerated over the lane graph and is **identical across arms**, and two properties
are reported before any cost is computed:

1. the number of topologically distinct candidates, and
2. the maximum separation between the candidates offered.

> **Rule.** A menu whose members differ by less than the robot's own width is reported as a
> **failed route-separation gate**, not as agreement between arms. (This is the gate the
> retracted `e4` campaign fails: two candidates 0.19 m apart, 15.05 m against 15.17 m.)

### M3.3 The runtime objective is a separate question

The budgeted rule above is a *stated* decision rule. Whether the deployed belief-space
objective — expected free energy with a risk, ambiguity, control and belief-inflated
obstacle term — also responds to availability is a distinct question, and the two must not be
conflated. Two constraints:

- **The visibility/ambiguity term is frozen method.** It is never reweighted to manufacture a
  route difference. Any effect must come from a structurally different channel.
- **All four cost terms are reported.** Availability enters this objective through *two*
  channels: the ambiguity term (small) and the obstacle term via the belief tube (large).
  Quoting only the ambiguity ratio misrepresents the objective by orders of magnitude.

## M4. Evaluation protocol

### M4.1 Prediction

- **Ground truth is the detector, not an oracle.** Every score is against real detector
  outcomes from real Gazebo captures.
- **Skill, not raw Brier.** A change that alters the base rate makes raw Brier move for free
  — the constant-prevalence arm "improved" by 0.044 across the reconfiguration while knowing
  only the base rate. Primary quantity is `1 − Brier/Brier_climatology` against each unit's
  own climatology.
- **Spatially blocked held-out.** On a dense pose grid a random split leaves held-out points
  centimetres from training points, so splits are leave-one-spatial-block-out. Units with no
  detections at all have undefined skill and are excluded by a rule fixed in advance.
- **Labelling threshold is declared and defended.** 0.25, the middle of the 0.05–0.5 plateau
  where both environments agree. At 0.01 the detector fires at 60 % of poses with no
  sight-line, which is why any dataset scored there is unusable for visibility labels.

### M4.2 Routes

Scored against real detector outcomes on a held-out heading partition: fields are fitted and
calibrated on one set of headings, routes are scored on a disjoint set. Reported per task,
per camera subset, per budget, with paired tests over tasks and the multiplicity correction
declared in advance. The experimental unit is the start–goal task.

### M4.3 Closed-loop **[REQUIRED, NOT YET RUN at four cameras]**

Preconditions, all of which must pass before runs are allocated:

1. **Route-separation gate** (M3.2) — a real menu, identical across arms.
2. **Heading-observability gate** (M2.2) — the planner's yaw variance must be measured, not
   sitting at its non-informative prior.
3. **Offline objective-sensitivity gate** — the runtime objective, evaluated on the real
   candidate menu at the belief state the robot actually carries, must rank at least two
   candidates differently across arms. No campaign is allocated on an objective that provably
   cannot discriminate.

Endpoints, declared in advance: time and distance driven with no accepted correction;
minimum clearance as a continuous statistic; goal attainment; physical contact. Termination on
goal, physical contact, or timeout only — a soft geometric graze is a statistic, not a
failure, because terminating on it produces a "safe halt" artefact.

Arms differ **only** in the observation-quality model. World, calibration, lanes, detector,
noise, seeds, controller and collision geometry are held identical and hashed into a manifest
before execution.

## M5. What is deliberately not varied

Held fixed so that a difference is attributable: conditional covariance `R` at its
commissioned value; the imaging term (measured null); the detector weights and threshold; the
lane network — the reconfiguration is chosen precisely so the driveable network is
bit-identical, which is what keeps observation modelling separate from obstacle avoidance.
Evaluation-only quantities — true pose, simulator depth buffer, oracle ray-cast visibility —
are never model inputs.
