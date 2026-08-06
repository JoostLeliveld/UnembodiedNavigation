# E6 external-log object-model check

This is permanent supporting evidence, not a publication-locked figure and not a closed-loop
result. It checks the CAD silhouette model on deployed driving logs that were not used by the
pixel-ground e1-e5 study.

## Result

On the 1,195 rows with recorded, effectively constant ground-truth yaw, applying the CAD
forward model reduced mean position error from 143.9 mm to 34.7 mm and changed mean
along-bearing error from -134.2 mm to +2.5 mm. The broader 1,421-row result, which includes
226 rows whose heading is inferred from the path tangent, is similar: 143.1 mm to 33.5 mm
and -133.0 mm to approximately zero.

The raw cross-bearing gate fired for cameras C and D. After accounting for the object model,
it no longer fired for C or D. This shows that the existing C/D constants are not identifiable
as camera calibration on these logs: their signal can be explained by silhouette geometry
coupled to the two fixed route headings.

It does **not** establish that every per-camera lateral correction is unnecessary. Cameras A
and B newly crossed the same heuristic gate after the model, and camera A is confounded with
one capture, one region and one heading. The next calibration study must vary camera, region
and yaw independently and evaluate a held-out stratum.

## Scope and consequences

- The logs contain only the deployed bottom-centre pixel, not bounding boxes. E6 validates
  the object model, not the candidate box-centre estimator, its 0.085 m plane or covariance.
- Only two recorded headings are available; the 45-degree diagonal is absent.
- The 226 tangent-derived rows are a weaker stratum and cannot carry the main conclusion.
- This is open-loop projection evidence. It authorizes no navigation or safety claim.
- The planned v2/v3/v4 closed-loop choice is blocked until the calibration components are
  identifiable under a yaw-diverse, route-disjoint design.

The complete generated output remains reproducible under ignored
`logs/studies/pixel_ground_path/e6_external_log_validation/`; hashes and restore information
are recorded in `provenance.json` here.
