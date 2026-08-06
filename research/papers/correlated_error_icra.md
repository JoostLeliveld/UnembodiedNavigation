# Correlated camera error and belief honesty

## Paper thesis

Persistent camera-specific error is repeatedly counted as fresh evidence by conventional
infrastructure-camera localization, producing a belief that is precise but wrong. A gated
calibration policy corrects resolvable outliers; per-camera residual flooring and a
leave-one-camera-out check prevent unearned confidence when calibration cannot remove the
error; a change monitor expires stale corrections.

## Included thesis claims

- C1: availability alone is insufficient.
- C3: honest observation modelling changes predicted belief.
- C4: navigation consequence, conditional on completion of `EXP-CL-CAL`.
- The calibration-lifecycle portion of C6.

This is not a GP paper, a generic fusion paper, or a claim that more cameras improve
localization.

## Evidence of record

- Up to 78 mm residual cross-bearing bias remains after deployed calibration.
- Projection amplification changes roughly 4.1 times across a footprint, but residual bias
  transfer dominates the remaining covariance problem.
- The conventional filter reports median NEES 4.22, 1.9 cm stated uncertainty versus
  5.3 cm RMSE, and 41.9 percent truth outside its stated 95 percent ellipse.
- Innovation gating rejects 0.2 percent and does not restore honesty; sharper per-camera
  covariance worsens median NEES to 5.11.
- The per-camera correlation floor plus leave-one-out check reduces outside-ellipse rate to
  3.3 percent while leaving RMSE essentially unchanged.
- Gated cross-bearing calibration reduces camera C bias from about 77 mm to 4 mm and
  held-out NEES from 8.51 to 1.06; ungated use harms a marginal camera.
- A stale correction becomes harmful at 0.25 degrees yaw drift; the change statistic
  detects at 0.1 degrees or 0.025 m translation.
- Availability and achievable precision choose different cameras on 15.7 percent of the
  reachable floor.

## Required final evidence

The only active scientific gate is the matched 30-run closed-loop campaign. Report
clean-goal rate, breaches/contacts, NEES/NIS calibration, path/time, and the full null if
navigation does not change. Do not open the source-comparison chapter before this package is
promoted.

## Scope

Gazebo-only; simulated detector imagery; one robot; 2-D position; controlled faults; four
nominally identical cameras with geometric and bias diversity; no formal safety or hardware
deployment guarantee.
