# Framing against the closest primary sources

| Work | Observation | Commissioning / supervision | Uncertainty | Multi-camera fusion | Future sensing prediction | Setting |
|---|---|---|---|---|---|---|
| [Bultmann et al.](https://arxiv.org/abs/2303.03797) | Robot model keypoints in RGB | Known robot model; camera calibration | Geometric reprojection formulation | Synchronized multi-view minimization | Not the evaluated contribution | Physical indoor robot |
| [Rabiee & Biswas](https://arxiv.org/abs/2306.16698) | Existing robot perception outputs | Sensing redundancy / consistency constraints | Empirical perception error distribution | Not an external-camera network study | Motivates risk-aware operation; not this installation route test | Mobile-robot perception |
| [Russell & Reale](https://arxiv.org/abs/1910.14215) | Learned regression / visual tracking / odometry | Supervised residuals; end-to-end filter training | Learned full multivariate covariance | Kalman-filter integration | No commissioned fixed-camera route prediction established in this work | Tracking and visual odometry experiments |
| [Zhang et al.](https://doi.org/10.36227/techrxiv.11663871.v5) | State-space measurements | Identifiability and sequential-data estimation | Noise covariance identification | General filtering problem | Not a future-camera-availability study | Estimation methodology |
| Current system | Frozen bbox-feature correction of metric reference position | Existing mean training + disjoint uncertainty/selection configurations | Conditional centred covariance + explicit mean and calibration | Fixed-Q replay; temporal ablation | Empirical joint hit/miss posterior averaging, one-route diagnostic | Simulation development only |

**Candidate contribution to establish:** commissioning an observation interface that predicts
usable localization information both at the current measurement and along future navigation
routes. Existing camera localization, learned covariance, Gaussian fusion and risk-aware
planning do not independently constitute novelty.

**Current defensible finding:** marginal uncertainty can score well while sequential coverage
changes materially with temporal dependence and observation history. Confidence conditioning
is competitive in this installation; the tested crop-statistics probe does not displace it.
This is a result to develop, not a demonstrated complete ICRA contribution.

Official ICRA 2027 requirements checked on 2026-09-05: [Call for Papers](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/).
The page states eight pages total, double-column PDF, double-anonymous review, and a
September 15, 2026 submission deadline. Its AI disclosure requirements also apply.
Formatting/submission work is paused at the user's request; no submission is being made.
