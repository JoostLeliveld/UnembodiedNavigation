# Next meeting: camera observation model before fusion or planning

## Meeting objective

Leave the meeting with agreement on three items:

1. confirm that the sensor output is one frozen YOLO bounding box and that only its downstream
   interpretation varies;
2. which residual representation and plots make their bias/noise intuitive;
3. which decision rule determines whether the next task is correction, conditional `R`, or
   rejection of an observation method.

The meeting does **not** ask the supervisors to choose a fusion rule, GP kernel, adaptive
filter, or planner treatment from speculative diagrams. Those follow from the measured
structure.

## Required evidence before building the deck

- One frozen YOLO bounding-box artifact and a registry of the downstream interpretations,
  with code path, available online inputs, output units, and oracle dependencies.
- A frozen capture/evaluation manifest covering five cameras, a controlled warehouse grid,
  and eight headings. The first field capture has one image per cell; repetitions are a
  separate follow-up because they answer a different question.
- One canonical residual table with a stable row identity for every attempted capture,
  including misses. The table must retain image-space and ground-plane residuals separately.
- A support audit showing how many positions, headings, repetitions, hits, and misses exist
  for every camera-method pair. Empty or weak cells must be visible rather than interpolated
  away.
- A clear label on every figure: single-sample field characterization, held-out result, or
  oracle/reference-only calculation.

The new capture is frozen for the first meeting question: 386 drivable floor positions, eight
headings, five cameras, and one RGB opportunity per camera-pose cell. It contains 15,440
attempted views with zero failed capture batches. The frozen detector returns 6,412 boxes
(41.5%) before any post-detection admission gate. Per-camera return counts are A 1,294, B
1,235, C 877, D 1,339, and E 1,667 out of 3,088 opportunities each. These are box-return
rates, not semantic visibility or runtime usability rates.

The capture can map the warehouse field and pooled histograms for the three deterministic box
interpretations, but it cannot split local mean bias from repeated-sampling covariance.
Present that limitation as the reason for the next repeat-panel decision. Do not smooth
missing cells or call one residual at a cell its mean.

## Slide-by-slide order

### 1. The question

**Title:** What does each camera actually tell the robot?

Show the warehouse, camera locations, robot, one YOLO output, and the chain from image output
to position observation. End with the problem: bias, random spread, missed detections, belief
uncertainty, and future availability are different quantities.

### 2. The observation interpretations

**Title:** The same YOLO box can imply different robot observations

Show one image, one YOLO bounding box, and one warehouse pose for raw box/IPM, fixed offset,
and analytic hull. Use the exact same box in every panel. Use the same colours and method names
throughout the deck. Mark oracle references visibly and keep them out of the method ranking.

Decision requested: approve the three box interpretations and keep alternative detectors out
of this experiment.

### 3. How the characterization was sampled

**Title:** Position, camera, heading, and repetition are controlled separately

Show the warehouse grid, five cameras, eight heading arrows at a sample point, repetitions,
and the fit/evaluation position split. State attempted captures and usable detections
separately.

### 4. How to read an error field

**Title:** An arrow shows systematic error; its background does not

Use one real RGB image and one deliberately simple camera-method example. Every arrow starts
at the commanded robot centre and ends at the interpretation implied by the box. On the
single-sample field, the arrow is an observed residual, not an estimated mean. This teaching
slide prevents later maps from being read as trajectories or camera pointing directions.

### 5. Warehouse error fields: one method across all cameras

**Title:** Does the same observation rule fail in the same way for every camera?

Use five small multiples with identical scale, colour range, cell support threshold, and
warehouse geometry. Repeat this slide for each observation method; do not overlay all methods
on one unreadable map.

Call out only measured patterns: smooth global direction, camera-relative radial error,
heading reversal, isolated low-support cells, or no visible structure.

### 6. Camera comparison: all methods for one camera

**Title:** Is the error caused by the camera or by how YOLO is interpreted?

For each camera, place method fields side by side at the same scale. This is the transpose of
slide 5 and is necessary to separate camera calibration from observation-method effects.

### 7. Heading and camera-relative components

**Title:** Is the displacement tied to robot orientation or camera geometry?

First show signed along-camera error versus the eight robot headings for all five cameras and
all three interpretations. Then show along-ray versus cross-ray residuals. This connects the
warehouse arrows to a physical mechanism: a fixed shift can remove a radial offset but cannot
remove a lateral box error.

### 8. The pooled histograms

**Title:** The whole-warehouse distribution is a mixture, not yet a noise model

Show `e_u`, `e_v`, along-ray, and cross-ray histograms for each camera-method pair. Use common
bins and mark mean, zero, median, and central intervals. State clearly that this pooled view
mixes locations and headings and cannot by itself justify a Gaussian `R`.

### 9. What this capture cannot estimate

**Title:** One residual at a cell is not a covariance

Show the hierarchy explicitly: this capture maps deterministic field structure and misses;
a predeclared repeated panel is needed for `R_hit`. Propose strata across camera, range,
viewing angle, occlusion, and image position. Freeze repeat counts after a pilot variance or
power calculation, not after seeing which method looks best.

Decision requested: which camera-method pairs deserve a repeated panel, and what conditional
variables must it cover?

### 10. Range, image position, and truncation

**Title:** The residual mechanism is testable

Show signed residual against heading, range, image coordinates, projected/detected box shape,
and truncation. Confidence may be shown as an explanatory variable but is not automatically
a planner input.

Decision requested: which mechanism should be tested by calibration or by changing the
observation interpretation?

### 11. Detection and miss maps

**Title:** Accuracy on hits and probability of a hit are different sensor properties

Show attempted versus usable detections over the warehouse for each camera. Place this beside,
not underneath, the residual result. A method that is accurate only because difficult samples
are rejected must show that availability cost.

### 12. Bias decision table

**Title:** The data selects the next model

Summarize each camera-method pair without prematurely ranking it:

| camera/method | bias structure | spread structure | tail shape | hit coverage | next action |
|---|---|---|---|---|---|
| pair | measured description | measured description | measured description | measured value | correct / retain / gate / reject |

Decision requested: approve one action per method. If correction is chosen, approve only a
realistically commissionable calibration and a spatially held-out evaluation.

### 12A. Actual-drive sanity check for the learned corrections

**Title:** Does the learned correction survive a continuous recorded route?

Show `09_learned_fixes_replayed/20_signed_bias_raw_vs_learned.png` immediately after the
field and distribution evidence. Raw, learned-linear, learned-neural and the oracle hull are stacked as four
time panels, each on its own y scale so both the large raw bias and the corrected residual
structure stay readable; the common-scale summary column on the right is where the
magnitudes actually compare. The same 385 raw camera readings come from one actual schema-5
Gazebo drive. The signed along-camera mean moves from −23.6 cm raw to −2.2 cm learned-linear, +0.8
cm learned-neural. Keep the words **offline replay** on both slides: neither corrected
reading stream steered the recorded trajectory or changed fusion.

Keep `21_along_ray_distributions.png` and `22_across_ray_distributions.png` as the
mechanism/backup slides. Separating
radial and lateral components makes the per-camera distributions readable and shows the raw
radial shift collapsing under the learned interpretations.

Do not select the linear or neural model from this one run. The decision requested is the
selection protocol: grouped spatial validation, a frozen model, then independent repeated
closed-loop drives.

### 13. What `R` will mean after the bias decision

**Title:** One covariance cannot mean four different things

Show the separation:

```text
actual hit -> R_hit -> filter update
future place -> q(s) and R_hit -> expected information -> planner
robot state uncertainty -> P
systematic field -> bias/calibration decision
```

Introduce the covariance ladder: global isotropic, per-camera isotropic, per-camera full,
geometry dependent, then spatial. Promise to stop at the simplest held-out model that is
calibrated and sharp.

### 14. A possible operational measurement method

**Title:** Pre- and post-update residuals may estimate `R_hit` without ground truth

Show pre-fit innovation `nu`, post-fit residual `mu`, and the projected belief-uncertainty
term. Explain that innovation variance is not `R` by itself. Propose the simplest
camera-specific covariance-matching estimator first and validate it offline against the
ground-truth characterization.

Decision requested: approve this as the first truth-free estimator test; defer a full
variational Bayesian filter unless the simple estimator fails for a reason Bayesian
uncertainty can address.

### 15. Where fusion will enter

**Title:** Cameras are combined before the planner scores the belief

Show one per-camera branch ending in state information
`q_i H_i^T R_hit,i^-1 H_i`, followed by either best-camera selection or justified information
addition. Mark shared camera errors and independence as an experimental gate.

### 16. What stays fixed in the planner

**Title:** The planner changes its observation assumption, not its machinery

List the frozen horizon, dynamics, constraints, goal and safety terms, action costs, belief
propagation, and tuning. Contrast constant observation information with the effective
state-dependent camera information. Explicitly exclude a direct visibility reward.

### 17. Work order and supervisor decisions

**Title:** Characterize, decide, estimate, aggregate, then plan

```text
1 characterization
2 bias gate
3 conditional R_hit
4 truth-free R_hit estimate
5 per-camera q(s,theta)
6 selection/fusion validation
7 unchanged-planner ablation
```

Record decisions, owners, required recaptures, acceptance tests, and what is deliberately
deferred. The next experiment starts only after the method registry and bias decision are
signed off.

## Figure rules

- Use centimetres for ground-plane effects and pixels only for image-space detector noise.
- Use identical scales when comparing cameras or methods.
- Never allow arrow length to auto-scale independently per panel.
- Show the number of samples supporting every cell; do not interpolate across unsupported
  warehouse regions without a separate model-labelled panel.
- Show unconditional hit coverage beside conditional error.
- Keep mean vectors and covariance ellipses visually distinct.
- A title states the finding; a caption states the dataset, split, camera, method, reference,
  admission rule, and sample count.
- Never label a plotted ground-truth residual as information available to the runtime.

## Definition of a successful meeting

The meeting succeeds if it produces a written decision on the method registry, the residual
views, the bias gate, and the exact next experiment. It does not need to produce a final
fusion rule or planner claim.
