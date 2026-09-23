# Current open thesis decisions

There is one open method decision: select the correction model and the covariance fitted to
that model's own whole-drive out-of-fold residuals.

The development comparison must decide:

- whether the raw-box MLP improves over the raw projection;
- whether the 16x16 visibility residual passes every predeclared selection gate;
- which planner-queryable `R_i(p, psi)` family is calibrated and sufficiently sharp;
- whether temporal or cross-camera residual dependence requires an augmented bias state.

The method is fixed, but these experiment-design values still need to be frozen before the
final fit or navigation campaign:

- which physical camera is removed in both removal conditions;
- the spatial kernel length scale and compact support radius for the planning-information
  field;
- the opportunity-support shrinkage constant and output grid spacing.

Choose these from coverage geometry or development data and record them before opening the
sealed audit. They are not additional model contributions and must not be selected from the
final navigation outcomes.

The following are not open:

- the planner uses a per-camera expected-information field fitted directly from all survey
  opportunities;
- admitted opportunities contribute matched out-of-fold runtime precision and
  non-admissions contribute zero;
- opportunity support is spatial and unsupported planning information shrinks towards zero;
- the measurement starts from the raw YOLO box-bottom projection;
- visual-hull and hull-equivalent observations are excluded;
- process covariance `Q_k` is frozen in the experiment configuration rather than inferred as
  a thesis contribution;
- navigation conditions share the runtime sensor model and compare intact, stale-removal and
  configuration-aware-removal planning-camera sets.

No historical study can reopen a locked choice. A change requires an explicit author decision
and a new contribution-lock version.
