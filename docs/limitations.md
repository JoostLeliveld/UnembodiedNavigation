# Limitations And Claim Guardrails

This document states the caveats that should appear in slides, reports, and papers based on the current repository.

## Main Caveats

### 1. The baseline is not true dead reckoning

The current primary comparison is:

- `efe1`
- `visibility_unaware_baseline`

The baseline still shares the same correction loop and state-estimation topology. It is visibility-unaware, not open-loop dead reckoning.

### 2. The main estimator is hybrid

In the primary image-detector path:

- `x, y` come from the external camera observation projected into BEV
- `theta` falls back to odometry

That means the repository currently supports:

`camera x,y + odometry theta`

It should not be described as fully camera-based pose estimation.

### 3. The GP is setup-specific

The GP visibility artifact is learned from the same simulated camera-detector setup used in the experiments. It should be described as:

- a learned visibility prior
- a pose-dependent detection-success model

It should **not** be overclaimed as a general occlusion model.

### 4. ET1 is the primary thesis planner path

The main claim path is ET1-based `efe1`. The repository still retains secondary modes `efe2`, `efer`, and `mpc`, but they should not be presented as equally central to the current milestone even though they now share the same cleaned planner implementation family.

### 5. The evaluator is still limited

The current evaluation layer is useful for milestone summaries and qualitative figures, but it is not yet strong enough for sweeping thesis-final claims.

## Claims That Are Safe

- the repo implements a controlled external-camera navigation comparison
- the planner can load and use a learned visibility field
- the two compared methods share the same simulator, camera, and state-estimation path
- the current repo can generate qualitative and milestone-grade summary figures

## Claims That Need Weakening

- “dead reckoning baseline”
- “fully camera-based online state estimation”
- “general occlusion-aware observation model”
- “all retained planner variants are equally validated”
- “evaluation is complete”

## Documentation Rule

If a statement ignores one of the caveats above, it should be rewritten or footnoted before it goes into:

- the root README
- package READMEs
- supervisor slides
- paper or thesis text
