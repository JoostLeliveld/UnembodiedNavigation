# Correlated camera error and belief honesty

## Paper thesis

Persistent structured projection error is repeatedly counted as fresh evidence by
conventional infrastructure-camera localization, producing a belief that is precise but
wrong. The present evidence supports per-camera residual flooring and a leave-one-camera-out
check as containment mechanisms. Whether the repeatable error should be attributed to camera
calibration, robot silhouette/yaw or their interaction is now an explicit open question.

## Included thesis claims

- C1: availability alone is insufficient.
- C3: honest observation modelling changes predicted belief.
- C4: navigation consequence, conditional on completion of `EXP-CL-CAL`.
- The calibration-lifecycle portion of C6.

This is not a GP paper, a generic fusion paper, or a claim that more cameras improve
localization.

## Evidence of record

- Up to 78 mm residual cross-bearing structure remains after deployed calibration on the
  historical routes; E6 shows that this is not identifiable as camera bias on those logs.
- Projection amplification changes roughly 4.1 times across a footprint, but residual bias
  transfer dominates the remaining covariance problem.
- The conventional filter reports median NEES 4.22, 1.9 cm stated uncertainty versus
  5.3 cm RMSE, and 41.9 percent truth outside its stated 95 percent ellipse.
- Innovation gating rejects 0.2 percent and does not restore honesty; sharper per-camera
  covariance worsens median NEES to 5.11.
- The per-camera correlation floor plus leave-one-out check reduces outside-ellipse rate to
  3.3 percent while leaving RMSE essentially unchanged.
- Historically, gated cross-bearing correction reduced camera C error on one held-out split
  and harmed a marginal camera. This is mechanism evidence, not an identified calibration
  policy after E6.
- On 1,195 external-log rows with recorded yaw, the CAD silhouette model reduces mean error
  from 143.9 mm to 34.7 mm and removes the previous C/D cross-bearing gate signal. Camera,
  region and yaw remain confounded.
- A stale correction becomes harmful at 0.25 degrees yaw drift; the change statistic
  detects at 0.1 degrees or 0.025 m translation.
- Availability and achievable precision choose different cameras on 15.7 percent of the
  reachable floor.

## Current decision gate

Do not launch the matched campaign yet. WS05 must first independently vary camera, spatial
region and yaw and determine whether any residual camera term transfers after the object
model. Only an identified held-out effect may become a causal arm. After that, the protocol
may choose one primary endpoint, generate the complete seed matrix and resolve current-world
versus July-field compatibility.

After those decisions and readiness pass, report clean-goal rate, breaches/contacts,
NEES/NIS calibration, correction acceptance/age, path/time, and the full null if navigation
does not change. A calibration-only contrast supports a calibration-consequence claim; it
does not establish closed-loop benefit for the complete correlation-floor/LOO method. Do
not open the source-comparison campaign before this package is promoted.

## Scope

Gazebo-only; simulated detector imagery; one robot; 2-D position; controlled faults; four
nominally identical cameras with geometric diversity and potentially confounded residual
structure; no formal safety or hardware-deployment guarantee.
