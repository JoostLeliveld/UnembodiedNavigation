# 05 — Overlap, selection, and conservative combination

## Show

1. `figures/overlap_handover_corridor.png`
2. `figures/overlap_gate_protocol.png`

## Say

“The red overlap contour shows where two or more camera views could see the
robot. That geometry is an opportunity to test agreement—not permission to
average observations. We collect synchronized D2 pairs, measure disagreement,
and pass a gate before enabling a handover or conservative combination.”

## Executed pilot evidence

- The long handover route produced three genuinely synchronized C/D pairs.
  They passed the strict 50 ms / 0.30 m gate: mean disagreement 0.247 m, no
  outliers. See
  [`../07_real_commissioning_execution/figures/04_actual_overlap_gate_C_D.png`](../07_real_commissioning_execution/figures/04_actual_overlap_gate_C_D.png).
- This is an accepted pilot edge, not a full campaign claim: a claimed general
  camera-pair edge still needs at least 30 held-out synchronized pairs and no
  more than 10% spatial outliers.
- Selection is the safe first combined system. Fusion is optional and follows
  only when consistency and covariance checks pass.

## Transition

“Once a source decision is justified, the unchanged belief/planner stack must
show that the new observation stream improves operation rather than merely
looking plausible on a map.”
