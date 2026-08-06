# Figure contract

| ID | Intended message | Evidence / required experiment |
|---|---|---|
| F01 | Persistent residuals are camera-specific and not explained by projection geometry alone. | EXP-BIAS, EXP-PROJ-AMP |
| F02 | Conventional updates become confidently wrong; residual flooring restores honest uncertainty. | EXP-BELIEF, EXP-RCOND |
| F03 | Calibrate resolvable outliers and leave marginal cameras raw. | EXP-BIAS, EXP-COMMISSION |
| F04 | A change statistic expires stale calibration before it becomes harmful. | EXP-DRIFT |
| F05 | Availability and achievable precision induce different camera choices. | EXP-PRECISION |
| F06 | Belief honesty changes clean-goal, breach, or calibration outcomes—or returns a documented null. | EXP-CL-CAL |
| F07 | Reliability sources differ in held-out calibration and error regime. | EXP-USABLE |
| F08 | Only models that discriminate routes receive closed-loop time. | EXP-USABLE |
| F09 | Accuracy is traded against commissioning, runtime, transfer, and adaptation. | EXP-USABLE, EXP-CAM-MGMT |
| F10 | Every method has an explicit failure case and fallback. | EXP-USABLE, EXP-CAM-MGMT |

Canonical paper figures live in `paper_artifacts/`. Scratch renderings remain ignored under
`logs/` and are deleted after promotion.
