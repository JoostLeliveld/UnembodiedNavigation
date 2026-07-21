# Four-camera warehouse — slide-by-slide walkthrough

Use these folders in order. The goal is to make the mechanism understandable
before discussing any paper-level evidence.

| Slide | Story beat | Visual to show |
| --- | --- | --- |
| [01](01_facility_and_live_streams/) | New facility | Four live camera views and the warehouse overview. |
| [02](02_day_zero_initialization/) | Before driving | One calibration-only prior per camera. |
| [03](03_uncertainty_aware_collection/) | One real data stream | Camera C’s original route, detector records, and pose covariance. |
| [04](04_per_camera_gp_learning/) | One dataset, several GP fits | The same Camera C records fitted by each method. |
| [05](05_overlap_selection_and_combination/) | Four individual GPs | How the four source-specific fields update differently. |
| [06](06_closed_loop_evaluation/) | Cameras working together | Coverage overlay, overlap region, and camera switching. |
| [07](07_real_commissioning_execution/) | Robot operation | A→B route, belief updates, and replanning target behaviour. |

Acts 03–07 should be added one clear visual at a time. A diagram that explains
the intended system must be labelled as an intended operation; a recorded plot
must name its actual data source.
