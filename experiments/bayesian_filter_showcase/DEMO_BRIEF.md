# Brief: build a demonstration that explains the filter

**Goal.** A step-by-step, visual explanation of what this filter does and why the result
happens. The owner of this repo has stated they cannot currently tell either from the
code or the figures. Treat "a reader understands the mechanism afterwards" as the
acceptance test, not "the figure is pretty".

This is an explainer, not a new experiment. Do not re-run Gazebo, do not change any
number, do not touch `research/registry.yaml`.

**Mandatory evidence label on every panel/caption:** this demo uses `BELIEF-V2`, a locked
mechanism study over 1,424 update steps from three named July captures projected by retired
v2. The 76.9/77/78 mm Camera C number is a historical signed cross-bearing bias, not current
camera error. Link `docs/localization_metrics.md`.

---

## 1. What the filter actually is

Do not go looking for a Kalman class. There isn't one. The filter is written inline in
`experiments/bayesian_filter_showcase/exp1_graceful_vs_trusting.py`, function `run_arm()`.

Ignore `reliability/toro_filter.py` — that is the Toro et al. **baseline**, a
constant-velocity filter with state `[x, y, vx, vy]` and no odometry. It is a comparison
method, not this.

The real thing is deliberately small:

| | |
|---|---|
| **state** | 2-D position only. `mean = [x, y]`, `cov` is 2×2. No velocity. No heading. |
| **predict** | dead reckoning. Add the odometry step, then grow the covariance by `PROCESS_SIGMA_PER_SQRT_M**2 * distance_travelled`. Uncertainty grows with distance driven, not with time. |
| **update** | each camera detection is a 2-D position measurement. Standard Kalman update with a per-arm measurement covariance `R`. |
| **A3/A4 only** | a *bank* of leave-one-camera-out beliefs runs alongside the main one. Camera c is judged by its innovation against a belief its own measurements never touched. |

That last row is the actual idea. Everything else is a textbook 2-D Kalman filter, and
the demo should say so plainly rather than dressing it up.

## 2. The seven arms differ in ONE thing

Identical data, identical prediction, identical everything except how a detection becomes
an update:

| arm | rule |
|---|---|
| A0 trust all | one fixed `R` for every camera and detection — the state of practice |
| A1 NIS gate | same `R`, plus a chi-square innovation gate: reject outliers |
| A2 sharp R | per-camera conditional covariance, tighter where the camera is believed good |
| A3 +LOO | A2 plus the leave-one-camera-out cross-check |
| A4 +floor | A3 plus a per-camera correlation floor — **the proposed method** |
| X1 floor only | ablation: the floor without the cross-check |
| X2 pooled | ablation: one shared floor instead of per-camera |

## 3. The result to explain

Calibrated median NEES is **1.386**. Anything above is overconfident.

| arm | median NEES | says 95%, gets | truth outside ellipse | RMSE |
|---|---:|---:|---:|---:|
| A0 | 4.22 | 58% | 41.9% | 53 mm |
| A1 | 4.13 | 58% | 42.1% | 53 mm |
| A2 | **5.11** | 56% | 43.6% | 48 mm |
| A3 | 4.55 | 61% | 38.6% | 52 mm |
| **A4** | **0.46** | **97%** | **3.3%** | 50 mm |
| X1 | 0.78 | 93% | 6.9% | 50 mm |
| X2 | 1.25 | 81% | 19.3% | 52 mm |

Four things a reader should leave understanding, in this order:

1. **Why A0 fails.** The bias is *constant*, so every detection from that camera repeats
   the same error. The filter treats each as independent fresh evidence and shrinks its
   covariance toward a wrong answer. Confidence grows while accuracy does not.
2. **Why gating (A1) cannot help.** A constant bias produces *small, consistent*
   innovations. It never looks like an outlier. A1 rejects 0.2% and changes nothing —
   4.13 against 4.22. This is the most counter-intuitive point and deserves the most care.
3. **Why sharper per-camera `R` (A2) makes it worse.** Tightening `R` on a biased camera
   tells the filter to trust the lie harder. NEES goes 4.22 → 5.11.
4. **Why the floor (A4) works.** A per-camera correlation floor stops repeated
   measurements from the same camera counting as independent. Outside-ellipse drops
   41.9% → 3.3% while **RMSE barely moves, 53 → 50 mm**. Honesty is not bought with
   accuracy.

Also state honestly: A4 lands at NEES 0.46 against an ideal 1.386, so it is now somewhat
*conservative*. It overshoots. Say so.

## 4. The historical input residual is recorded, not injected

Under the retired v2 projection on these route/yaw-confounded captures, camera C carries a
**recorded +76.9 mm signed lateral bias**. It is not a current camera property. That repeated
residual is invisible to camera C alone — it is only
detectable because three other cameras disagree, which is what makes this a
multi-camera mechanism rather than a filtering trick. The demo should make both the mechanism
and historical-input scope visible.

## 5. What would make a good demonstration

The owner mentioned being inspired by a notebook. **Ask which one before designing** —
there is no notebook in this repo, and the reference is external. If it is Labbe's
*Kalman and Bayesian Filters in Python*, the house style is: one idea per cell, a picture
immediately after each, and the maths introduced only once the picture makes it obvious.

Strong candidates, roughly in order of explanatory value:

- **A single detection, drawn.** Prior ellipse, measurement with its `R` ellipse, posterior
  ellipse. Three shapes on one axis. Most people have never seen the update as a picture.
- **The same camera measured ten times.** Show the covariance shrinking each time under
  A0, with truth sitting still outside it. This is the whole failure in one animation or
  strip of panels.
- **Innovation histogram, A0 vs A4.** Shows *why* the gate cannot see the bias: the
  innovations are small and consistent, not large and rare.
- **A trajectory with the stated 95% ellipse drawn along it**, A0 beside A4, truth
  overlaid. The reader should be able to count the escapes by eye.
- **The leave-one-out idea, drawn.** Four beliefs, one per excluded camera, and the
  disagreement that exposes camera C.

Prefer a small honest set over a large one.

## 6. Data and constraints

- Captures and detections load through `rcond_common` (imported as `rc` in the script);
  1,424 detections across three captures, four cameras.
- Ground truth is **evaluation-only** — it scores the arms and never enters a filter.
  Any demo that feeds truth into an estimator is wrong.
- Never hand-roll metrics: `scripts/shared/metrics.py` is the shared library.
- Figures go to `logs/studies/bayesian_filter_showcase/<exp>/`, then
  `python3 scripts/research/promote_figures.py` publishes them to `figures/EXP-BELIEF/`.
- Suite is `python3 -m pytest tests/ -q` from the repo root; it currently passes 950.

## 7. Scope

This figure is `F02` in `research/08_figures.md`, and F02 is the paper's core result — the
overconfidence claim and its containment. It is marked READY with no canonical renderer.
A demonstration that makes the mechanism legible is directly on the paper's critical path.

Do not extend into the pixel→ground projection work, the reliability-source benchmark, or
the closed-loop campaign. They are separate and currently blocked.
