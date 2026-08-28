# Can the camera network measure its own noise?

**Question.** Commissioning measures each camera's noise once, in a lab exercise against ground
truth, on a stationary robot. Deployments drift: a camera is knocked, a rack is restocked, a
lens fogs. Can the network re-measure itself from an ordinary drive, with no ground truth?

**Serves:** the operational-covariance contribution of the fusion paper
(`../fusion_on_fixed_routes/`). Outputs in `logs/studies/learned_measurement_covariance/`.

## The two scripts

`estimate_r.py` — the estimator. At each instant where several cameras report, take each
camera's distance from the average of them all. What the cameras get wrong *together* cancels
in that subtraction, and so does the truth — which is why no ground truth is needed. What
remains is that camera's own noise, mixed with a known fraction of everybody else's:

    spread of (camera c − the average)  =  (1−1/N)² × noise of c  +  (1/N²) × sum of the rest

One equation per camera per instant, solved for all cameras at once. With `--write=<file>` it
emits a calibration artifact in the same shape as the commissioning one, so pointing
`manager_commissioned_calibration_path` at it is the whole of "recommission the network".

    python3 estimate_r.py --write=out.json <drive dir> [<drive dir> ...]

`measure_delay.py` — separates the sensor's own error from the pipeline's delay. Every earlier
measurement compared a reading to the truth at *logging* time, which is later than the camera's
capture time by the whole detector-and-manager delay. That delay is identical on every camera,
so it looks exactly like measurement error and gets charged to each camera's noise. Since
2026-08-27 the capture time is logged (`obs_stamp` in `fusion_observations.csv`) and the two
can be told apart.

    python3 measure_delay.py <drive dir> [<drive dir> ...]

## What it cannot do, and this is permanent

A lean shared by every camera is invisible to it. Differences cancel a common error exactly,
which is the same property that makes the estimator work. Measuring that part still needs a
survey, or the robot standing somewhere known. The estimator reports each camera's lean away
from the network mean so the visible part is at least stated.
