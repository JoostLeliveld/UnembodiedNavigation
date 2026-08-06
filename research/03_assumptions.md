# Assumptions register

Assumption state is `ACCEPTED`, `TESTED`, or `DEFERRED`. The registry contains the current
state; this document gives the scientific consequence.

| ID | Exact statement | Why needed / plausible | Sensitivity or justification | Consequence if violated |
|---|---|---|---|---|
| A01 | Commissioned calibration is available; drift is bounded or monitored. | Fixed cameras are normally commissioned once. | Controlled yaw/translation drift ladder. | Projection bias grows and stale corrections can become harmful. |
| A02 | The operational floor is locally planar. | The camera produces a 2-D ground-point measurement. | Projection-amplification study; uneven floors are deferred. | Homography and ground covariance are biased. |
| A03 | Intrinsics, optics, resolution, and timing are fixed within an arm. | Ensures method is the independent variable. | Freeze all camera/runtime configurations in manifests. | Results confound estimator with hardware. |
| A04 | Static mapped depth and live sensed depth have distinct provenance. | Their cost and staleness differ materially. | Separate benchmark labels and stale-layout arm. | “Depth” becomes operationally ambiguous. |
| A05 | Unknown depth has an explicit conservative fallback. | Real depth maps contain missing cells. | Missing-depth ablation: unavailable, FOV fallback, or conservative prior. | Raycast can silently predict visibility through unknown space. |
| A06 | Static and dynamic occlusion are separate regimes. | Shelves differ from people, pallets, and forklifts. | Static benchmark first; injected dynamic occluder later. | Aggregate results hide a method's true failure mode. |
| A07 | Lighting and real-image transfer are outside current claims. | Detector and worlds are simulated. | State as a limitation; make no hardware claim. | The learned estimator may fail under image-domain shift. |
| A08 | Detector weights and threshold are frozen within comparisons. | Quality sources must see the same perception process. | Hash model and record confidence threshold. | Source ranking can be caused by detector changes. |
| A09 | Robot dimensions and appearance are fixed. | Visibility labels depend on the target. | Record mesh, footprint, and silhouette statistic. | Transfer to another robot is unsupported. |
| A10 | There is one robot and no association ambiguity. | Matches all current captures and campaigns. | Explicit scope limit. | `p_use` must include target association. |
| A11 | Heading is odometry-backed; cameras correct 2-D position. | Matches the locked estimator state. | Keep heading noise fixed across arms. | Camera heading observability becomes a new research problem. |
| A12 | Synchronization and latency are measured operational inputs. | Timestamp error changes residuals and fusion. | Timing/coverage audit and dropout/latency arm. | Residual floors mix geometry with stale-state error. |
| A13 | Errors may persist within and correlate across cameras. | Repeated views do not create independent evidence. | NEES, leave-one-out, and correlation-floor ablations. | Independent fusion becomes overconfident. |
| A14 | Evidence is simulation-only. | Enables controlled faults and exact evaluation truth. | Explicitly make no real-hardware claim. | External validity remains unproven. |
| A15 | Evaluation truth is unavailable operationally. | Prevents leakage into deployed policies. | Validator compares operational and evaluation-only interfaces. | Results become oracle-assisted and invalid. |
| A16 | Current cameras vary geometrically and in measured bias, not optics. | Hardware is nominally identical. | Report positions, occlusions, handovers, and bias only. | Four optical-archetype claims would be false. |

## Source-benchmark freeze checklist

Before `EXP-USABLE` becomes active, freeze the world splits, route alternatives, detector
hash and threshold, calibration, robot geometry, candidate-pose grid, operational feature
contract, evaluation labels, random seeds, and missing-data fallback. Noise is applied at
the shared measurement interface—not tuned separately per method—and uses identical seeds
for paired comparisons.
