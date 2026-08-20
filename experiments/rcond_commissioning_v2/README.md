# Rcond commissioning v2 — repeated-pose Gazebo campaign

This campaign demonstrates the measurement model that the previous per-cell
inverse-Wishart loop could not identify:

```text
reading = truth + predictable bias(geometry, heading)
                  + persistent site/session effect
                  + independent repeatability noise
```

It is an observation commissioning run, not a navigation campaign. Ground truth
is used offline to learn and score the observation model and is never an online
camera/filter input.

The capture uses the AWS development world and the retrained AWS keypoint model.
Every nominal site/heading receives balanced pose/yaw micro-jitter, and the same
plan is repeated after a full Gazebo restart. All repeats and headings from one
spatial anchor share a split. The final session is a session holdout.

## Pilot

```bash
bash experiments/rcond_commissioning_v2/run_pilot.sh
python3 experiments/rcond_commissioning_v2/analyse_campaign.py \
  --root logs/studies/rcond_commissioning_v2/pilot
```

The pilot checks transport, grouping, detector inference and analysis. It is too
small for a paper claim.

Latest local pilot (`logs/studies/rcond_commissioning_v2/pilot`) completed two
independent Gazebo restarts with 144 qualified readings. From 48 session-0
training readings it learned a 1.002 px posterior scale and froze the
conservative 1.209 px posterior-95% value before looking at session 1. Geometry
propagation then scored 1.006x truth/claimed scale on that restart holdout (1.00
ideal). The alternative global ground-space bias/floor diagnostic failed
unseen-space coverage (12.5%), so it is explicitly not planner-ready. This is
the intended distinction: learn noise in image space and let
`J(x) R_uv J(x)^T` create spatial ellipses; do not infer an arbitrary
ground-space R map from repeatability alone.

Meeting figures are generated under the pilot's `analysis/` directory:
`fig_geometry_r_pilot` is the accepted construction and `fig_rcond_pilot` is
the rejected global-floor diagnostic.

## Full campaign

Use `run_full.sh`: 10×6 spatial anchors, four headings, five repeats and three
independent Gazebo sessions. Session 2 and the spatial validation blocks remain
untouched by model fitting. Run it only after the pilot report is valid.

Simulation caveat: micro-jitter samples local rendering/detector sensitivity but
does not reproduce real camera sensor noise. A hardware claim still requires
real repeated images across independent sessions and lighting conditions.
