# WS01 — claims and method taxonomy

## Objective

Turn SQ1-SQ4 and C1-C6 into a stable claim architecture for two distinct outputs: the
current correlated-error/belief-honesty paper and the later reliability-source chapter.
Define what is representation, estimation, planning-time use, and camera management so
later chats cannot mix them.

## Ownership

Writable:

- `research/02_claims.md`
- `research/05_method_taxonomy.md`

Read-only:

- `research/registry.yaml`, `research/01_questions.md`
- `research/papers/correlated_error_icra.md`
- `research/papers/reliability_source_comparison.md`
- `experiments/usable_observation/README.md`
- `experiments/bayesian_filter_showcase/README.md`
- `experiments/closed_loop_calibration/README.md`
- locked `RESULTS.md` and summary JSON files relevant to C1-C6

Do not edit the registry, generated status, experiment code/configs, paper scopes, or
evidence. Propose those changes in the handback.

## Requirements

- SQ1-SQ4 are immutable.
- Preserve C1-C6 IDs; refine prose without inventing a competing claim system.
- Every claim needs a falsifiable endpoint, evidence class, scope, and explicit non-claim.
- The current paper must not depend on completion of the source benchmark.
- Represent observation quality with separate `p_use`, conditional covariance, persistent
  bias/correlation floor, epistemic support, and freshness/health.
- Constant, distance, FOV/range, depth/raycast, GP, hybrid and gated DL are estimator arms.
- Nearest camera, maximum `p_use`, achievable precision, hysteresis and conservative fusion
  are management policies using frozen fields, not estimator arms.
- Candidate-pose estimators may use only future-pose-available operational information.
  Instantaneous detector outcome/confidence and recent observations belong to management.
- Current cameras support claims about viewpoint geometry, occlusion, overlap role and
  measured bias—not optical diversity, vendors, hardware transfer or generic warehouses.

## Known evidence and tensions

- Existing evidence strongly supports camera-specific persistent error and correlated-error
  overconfidence.
- Existing GP results are null/negative against simple geometry on held-out data; preserve
  that rather than writing a GP-superiority claim.
- The active closed-loop comparison changes calibration only. It cannot establish that the
  whole correlation-floor/LOO method improves navigation.
- The existing usable-observation study is single-camera and nearly entirely a `p_det`
  benchmark because `p_qual` is close to one.

## Deliverables

1. A C1-C6 table with endpoint, evidence, scope and prohibited interpretation.
2. Separate subquestion trees for paper A and chapter B.
3. A taxonomy table for every source family: estimand, legal inputs, commissioning,
   transfer assumption, runtime, failure, fallback and current evidence.
4. A crisp representation/estimation/planning/management boundary.
5. A handback listing proposed changes to the two paper scope files and registry.

## Acceptance criteria

- Every claim is answerable by at least one registered experiment or explicitly open gate.
- No navigation or safety claim is inferred from frame-level/offline prediction alone.
- No camera-selection policy appears among quality-source estimators.
- GP and other null results remain visible.
- No optical-diversity, hardware or broad warehouse-generalization language survives.

## Paste-ready prompt

```text
Work only on the claim architecture and method taxonomy in:
/home/joostleliveld/Thesis/UnembodiedNavigation

Read research/registry.yaml, research/01_questions.md, research/02_claims.md,
research/05_method_taxonomy.md, both files under research/papers/,
experiments/usable_observation/README.md,
experiments/bayesian_filter_showcase/README.md,
experiments/closed_loop_calibration/README.md, and the relevant locked RESULTS.md and
summary JSON evidence.

You may edit only:
- research/02_claims.md
- research/05_method_taxonomy.md

Do not edit research/registry.yaml or generated research/STATUS.md. Do not run or implement
experiments.

Keep SQ1-SQ4 and C1-C6. Produce a falsifiable claim/evidence/scope/non-claim table and split
the current correlated-error paper from the later reliability-source chapter. Explicitly
separate: (1) representation using p_use, R_cond, persistent bias/correlation floor,
epistemic support and health; (2) estimation by constant/distance/FOV/depth/GP/hybrid/DL;
(3) planner use via expected hit/miss belief; and (4) downstream camera management via
nearest/max-p_use/achievable-precision/hysteresis/conservative fusion with frozen fields.

Retain existing GP null results. Restrict camera-diversity claims to geometry, occlusion,
overlap role and measured bias: the four cameras are optically nominally identical. Return
files changed, unresolved scientific decisions, and proposed registry/paper-scope changes.
```
