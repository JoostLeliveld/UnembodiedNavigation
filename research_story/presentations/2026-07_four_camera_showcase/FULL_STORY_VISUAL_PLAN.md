# Four-camera visual sequence

The presentation should explain one mechanism in a readable order, not try to
prove every paper claim at once.

1. **Facility:** show the big warehouse and four live streams.
2. **Initialization:** show one day-zero GP prior per source.
3. **Collection:** take one camera on one original route and show exactly what
   becomes a GP record: projected pose, pose covariance, detector score, hit,
   and miss.
4. **Fitting:** reuse that exact record set in four GP fitting variants. Keep
   the input markers fixed across plots so the audience sees only the fitting
   choice change.
5. **Source specificity:** place the four final source-specific fields on the
   same floor plan. Do not blur them into one map.
6. **Combination:** overlay the four camera footprints and an enlarged overlap
   corridor. Show the intended source-switch timeline along a route.
7. **Operation:** draw the target A→B plan with camera-source changes,
   belief corrections, and replanning. Label this final visual as the intended
   operational behaviour until repeated closed-loop evidence exists.

Use a recorded plot whenever data exists; use a simple labelled system visual
when explaining the target operation. Do not use generic workflow cartoons or
unlabelled “result” charts.
