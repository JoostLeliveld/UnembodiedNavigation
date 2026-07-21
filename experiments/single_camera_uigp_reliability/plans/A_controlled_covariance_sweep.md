# Experiment A — controlled covariance-fidelity sweep (ch.03)

**Status label on every output: "controlled ablation, not operational evidence."**
This proves the uncertain-input mechanism behaves correctly when the true input
uncertainty is *known*. It is not the paper headline (that is Experiment B).

## Question
When the reported training-input covariance is scaled `Σ_reported = α·Σ_true`,
does U5 (expected-kernel) degrade more gracefully than the point-input GPs, and
at what α does it separate from U1/U3?

## Construction (real trajectory preferred over synthetic)
- Take a REAL `warehouse_aws` detector run (Experiment B's capture, or a short
  dedicated pass). Use evaluation-only GT to define `Σ_true` residual scale.
- Build training locations `μ_t = x_t^GT + ε_t`, `ε_t ~ N(0, Σ_true)`; feed the
  GP `Σ_reported = α·Σ_true`.
- Regimes: accurate `~0.01²`, moderate `~0.10²`, anisotropic (rotated `0.30²/0.05²`).
- α ∈ {0, 0.5, 1, 2, 4, 8} (α=0 ⇒ ignore uncertainty ⇒ collapses to U1).

## Methods
U0–U5 (+ U6 GT ceiling, evaluation-only). U5 = `expected_kernel`; U1 = `naive`.
Reuse `fit_belief_aware_gp.py` modes directly — do NOT reimplement the kernel.

## Metrics (via `scripts/shared/metrics.py`)
Held-out Brier, Bernoulli NLL, ECE, AUROC, AUPRC, false-trust rate at τ∈{0.8,0.9,0.95};
if a synthetic latent field exists, integrated squared error; correlation between
GP posterior σ and actual held-out error.

## Key figures
1. noisy training points with covariance ellipses;
2. U1 vs U3 vs U5 mean maps; 3. U1/U3/U5 posterior-σ maps;
4. **error-vs-α curve** (the headline of A) — U5's curve degrades most gracefully;
5. the α at which U5 separates from U1/U3 = the operating-regime statement.

## Reuse
`logs/studies/optionA_commissioning/exp1_synthetic_gp` harness already produces
setup/prediction-map/metrics figures; **extend** it to the U1/U3/U5 + σ panel
(ch.03 README says exp1 extends to this). Outputs →
`logs/studies/single_camera_uigp_reliability/expA_covariance_sweep/`.

## Gate
Mechanism correct (α=0 ≡ U1; monotone graceful degradation of U5). This gate
being green does NOT license any claim — it only unlocks Experiment B.
