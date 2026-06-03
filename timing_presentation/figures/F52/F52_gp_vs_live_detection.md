# F52 - GP Versus Live Detection

- Figure: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F52/F52_gp_vs_live_detection.png`
- PDF: `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F52/F52_gp_vs_live_detection.pdf`

This diagnostic checks whether the planner-facing GP reliability agrees with the live YOLO detector during F50. A mismatch matters because C2 can only execute the intended visibility-aware behavior if the route that looks reliable to the GP is also observable by the runtime detector.

Interpretation from F50:

- C2 seed 1 is the desired execution case: detector rate is 1.00, mean GP reliability is high, and the robot reaches the goal.
- C2 seed 2 is not a GP-overconfidence failure. The GP drops to low reliability on the route and live detections disappear there. This means the visibility-aware global route still allowed a segment that becomes camera-dark in execution; the fix is to make the route-choice/global objective avoid that segment more reliably or choose a task where the visible detour remains visible end-to-end.
- C2 seed 0 is a different failure: detector rate is 1.00 and mean GP reliability is high, but the run collides early. This points to local tracking/no-go clearance rather than perception.

Next fixes should split by failure type: improve local tracking clearance for the C2 seed 0 style failure, and improve global route scoring/reliability gating for the C2 seed 2 style failure. Tuning ambiguity weight alone is too blunt because the failures have different causes.
