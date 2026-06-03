# F51 - F50 Multi-Seed Diagnosis

Files:
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F51/F51_f50_seed_diagnosis.png`
- `/home/joostleliveld/Thesis/UnembodiedNavigation/timing_presentation/figures/F51/F51_f50_seed_diagnosis.pdf`

Conclusion: F49 was a useful smoke success but F50 is not robust paper evidence. C1 fails in all completed seeds, which is consistent with the risky baseline, but C2 only reaches in seed 1. C2 seed 0 collides early and seed 2 times out after long stale visual-correction periods. The route-choice layer can produce the intended visibility-aware route, but the execution layer is still dominated by first global-solve latency, local max-iteration events, and stale visual updates during tracking.

The decisive split is detector availability. The successful C2 seed keeps live detections available (`det rate=1.00`, max planner pixel-correction age < 1 s). The failed/stalled C2 seed loses detections for most of the run (`det rate=0.33`, max planner pixel-correction age ≈ 58.5 s). Therefore the next fix should not be more ambiguity weight. It should verify whether the GP-preferred route is actually observable by the runtime detector, and then either refit/adjust the GP/scene or choose a route-choice task whose learned reliability agrees with live YOLO evidence.

Next fix should target the runtime architecture, not the ambiguity weight: make the global plan a preflight or cached solve, tighten failure classification for repeated safe-stops, and improve local tracking robustness before treating AWS B1 as evidence.
