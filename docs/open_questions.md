# Current open thesis decisions

There is one open method decision: select the correction model and the covariance fitted to
that model's own whole-drive out-of-fold residuals.

The development comparison must decide:

- whether the raw-box MLP improves over the raw projection;
- whether the 16x16 visibility residual passes every predeclared selection gate;
- which planner-queryable `R_i(p, psi)` family is calibrated and sufficiently sharp;
- whether temporal or cross-camera residual dependence requires an augmented bias state.

The following are not open:

- availability is `q_i(p)`, with camera identity and 2-D position as its only query inputs;
- surveyed headings are pooled for availability;
- the measurement starts from the raw YOLO box-bottom projection;
- visual-hull and hull-equivalent observations are excluded;
- process covariance `Q_k` is frozen in the experiment configuration rather than inferred as
  a thesis contribution;
- navigation conditions share the runtime sensor model and differ only in future `q` and `R`
  forecasts.

No historical study can reopen a locked choice. A change requires an explicit author decision
and a new contribution-lock version.
