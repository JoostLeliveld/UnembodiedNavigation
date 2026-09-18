# The covariance ladder R0 – R2

What each rung is, stated precisely enough to put in the thesis, plus the two
constants that must be selected rather than hand-set and the one mechanism that
has to change when the residual population moves from drives to static poses.

Scope: correction candidates **C0** (raw, no correction) and **C3** (box MLP),
covariance models **R0**, **R1**, **R2**. R3 and R4 exist in code but are out of
scope.

## The shared quantity

Every rung consumes the same residual population: the corrected measurement
error of ONE frozen correction, in ground-plane metres,

    r_{i,n} = z̃_{i,n} − H_s x_n^ref

and returns a 2×2 covariance for the current observation. Per `AGENTS.md`, each
correction candidate gets its own covariance fit; one candidate's `R` is never
reused for another.

**Residuals are not recentred before covariance fitting.** All three rungs
estimate the second moment about zero, `E[r rᵀ]`, because the downstream filter
assumes a zero-mean measurement error. If a non-zero residual mean survives the
correction, this raises the covariance handed to the filter and so reduces that
measurement's influence, instead of hiding the bias by subtracting a local
sample mean. It does NOT make the zero-mean Gaussian model correctly specified —
it is a deliberate choice about which failure is safer, not a fix.

All three end with an eigenvalue floor, applied after the rung's own arithmetic:

    Σ ← V max(Λ, 1e-6) Vᵀ,   Σ = ½(Σ + Σᵀ) eigendecomposed as V Λ Vᵀ

1e-6 m² is a 1 mm standard deviation. It exists so a degenerate local sample
cannot produce a singular `R` and an unbounded Kalman gain.

## R0 — one covariance for the installation

    R_0 = (1/N) Σ_n r_n r_nᵀ           pooled over every camera and position

Three free parameters. Conditions on nothing. It is the floor of the ladder: a
richer rung that cannot beat R0 has not earned its parameters.

Answers: *does localization uncertainty differ at all?*

## R1 — one covariance per camera

    R_{1,i} = (1/N_i) Σ_{n: cam=i} r_{i,n} r_{i,n}ᵀ

Fifteen free parameters, 3 per camera. Conditions on camera identity only.

Answers: *does camera identity explain the uncertainty?*

## R2 — per-camera spatial covariance with shrinkage

Kernel-weighted local second moment, shrunk toward that camera's R1. Retain the
K nearest covariance-fit residuals for camera `i`:

    N_i(p) = KNN_K(p; {p_{i,n}}_{n ∈ D_R})

raw kernel weights

    w̃_{i,n}(p) = exp( −‖p − p_{i,n}‖² / (2 ℓ²) )

a grouping normalisation (see below), then

    S_i(p) = Σ_{n ∈ N_i(p)} w_{i,n}(p) · r_{i,n} r_{i,n}ᵀ

    R_{2,i}(p) = ( S_i(p) + λ R_{1,i} ) / ( Σ_n w_{i,n}(p) + λ )

followed by the eigenvalue floor. Current implementation values: `K = 120`,
`ℓ = 0.75 m`, `λ = 4.0`.

Answers: *is there additional repeatable variation within one camera's view?*

### Call it kernel-weighted shrinkage, not "Bayesian"

The estimator is a kernel-weighted second moment shrunk toward a per-camera
prior. It CAN be given a Bayesian reading — for zero-mean Gaussian residuals,
put an inverse-Wishart prior on the local covariance with prior mean `R_{1,i}`
and prior strength `λ`, and treat the kernel weights as fractional observation
counts; the posterior mean is exactly the expression above. If the thesis wants
that claim it must derive it, including why fractional weights may stand in for
counts. Otherwise the honest name is **kernel-weighted shrinkage covariance**.

### The behaviour that suits a fixed installation

Where local evidence is dense, `Σw ≫ λ` and the local residual structure
dominates. Where it is sparse, `Σw ≪ λ` and `R_{2,i}(p) → R_{1,i}`: the rung
degrades gracefully to R1 rather than fitting noise from two points.

That degradation is not hypothetical here. Measured on the current `D_R`
partition, the fraction of each camera's own admitting floor holding fewer than
5 residuals within 2 m:

    camera A  21.6%     B  43.6%     C  34.0%     D  25.5%     E  22.7%

Camera B is below usable local support on 44% of the ground it can see. Over
that ground R2 IS R1, by construction. Re-measure with
`experiments/warehouse_v2_sketches/plot_local_residual_support.py`.

## ℓ and λ must be selected, not hand-set

`ℓ = 0.75 m` and `λ = 4.0` decide exactly how spatial R2 really is: together
they set the evidence mass needed before local structure overrides the
camera-level prior. They are currently implementation defaults with no
derivation, which is the same species of defect as an undeclared planner weight.

Select them on `D_dev`, before the final audit and before any navigation run,
over a small predeclared grid, with held-out Gaussian NLL as the primary
criterion and containment (50/90/95) as a calibration check. Record the grid and
the selected values.

This is legitimate model selection and is NOT the thing ruled out elsewhere in
this repo: choosing a planner weight by which route it produces optimises the
outcome the weight is judged by. Here the criterion (held-out likelihood) is
independent of the quantity the thesis reports (navigation performance).

## The grouping unit must change with the dataset

The shipped implementation normalises kernel weights **per drive**: if residual
`n` belongs to drive `g(n)`,

    w_{i,n}(p) = w̃_{i,n}(p) / Σ_{m: g(m)=g(n)} w̃_{i,m}(p)

so every represented drive contributes equal total mass regardless of how many
samples it supplies locally.

That was the right defence for the population it was built on. Verified: the
current `box_mlp_residual_population.npz` holds **21 642 residuals from 18
driving laps** (`audit_settle_lap6_r3`, `fit_settle_lap2_r2`, …), and within one
lap consecutive residuals are a median 0.465 m apart — densely autocorrelated,
exactly what per-drive normalisation exists to stop from dominating a
neighbourhood.

**It becomes the wrong unit once `R` is fitted to the static reference
dataset.** There are no drives there. The independent unit is the reference
pose, and the normalisation must follow:

    w_{i,n}(p) = w̃_{i,n}(p) / Σ_{m: pose(m)=pose(n)} w̃_{i,m}(p)

so one static pose cannot dominate a neighbourhood merely by contributing more
frames or headings. If captures at a pose are genuinely independent, an
alternative is no normalisation at all and one residual per camera opportunity.
Choose deliberately; do not carry the drive mechanism over by inertia.

Truncation to the K nearest residuals happens BEFORE normalisation.

## One field serves runtime and planning

If R2 is selected, the same fitted function serves both:

    R_i^run(p)  = R_{2,i}(p)      queried at the realised position
    R_i^plan(p) = R_{2,i}(p)      queried at a predicted future position

The distinction is only where it is evaluated. No separate planning covariance
is needed.

## A geometry rung is optional, and depends on the claim

`R_geom(d, α, i)` — shared low-capacity coefficients in range and viewing angle,
with per-camera intercepts — would sit between R1 and R2 and is more
data-efficient than either where support is thin.

Add it only if the thesis wants to claim the learned spatial field captures
something BEYOND conventional camera geometry. If the claim is spatial system
identification of a fixed installation, R0 → R1 → R2 already asks the three
questions that matter and a fourth rung is another branch to defend.

## Runtime consequence

R2 needs the residual sample at inference time, so the commissioned package
ships the residuals themselves, not only fitted parameters. R0 and R1 ship a
matrix. This is a deployment difference worth stating.
