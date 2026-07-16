# 06 — Closed-loop evaluation

## Show

`figures/closed_loop_evaluation_protocol.png`

## Say

“The final comparison is matched and operational: same route seeds and world,
then compare a fixed source, a simple score policy, a reliability-aware manager,
and conservative combination where the D2 gate allows it. We report the whole
distribution of goals, breaches, recovery, belief error, covariance honesty,
and handover behavior.”

## Executed pilot evidence

- Shadow replay has been run over 57 operational replay frames with the four
  newly fitted providers and the hysteretic handover policy.
- At the configured 0.45 trust release threshold it released zero corrections:
  the safe-defer path worked as designed. See
  [`../07_real_commissioning_execution/figures/05_actual_algorithm_execution.png`](../07_real_commissioning_execution/figures/05_actual_algorithm_execution.png).
- This remains a safety/integration result, not a closed-loop improvement
  claim. Matched, route-disjoint live corrections and evaluation stay required.

## Closing line

“Four cameras give coverage; uncertainty-aware commissioning earns credible
camera-specific maps; overlap-gated combination turns that evidence into a safer
belief correction.”
