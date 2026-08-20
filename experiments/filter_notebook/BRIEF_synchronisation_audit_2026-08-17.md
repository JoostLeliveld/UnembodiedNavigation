# Brief: synchronisation audit of the AWS single-camera captures

**Written** 2026-08-17. **For** a fresh session with no history of this work.
**Read this whole file before touching anything.** It exists because the obvious
first moves — re-running a capture, widening an association tolerance, "fixing"
the grid snap — would destroy the evidence base for the defence.

---

## 1. What you are auditing, and what you are not

You are auditing **time synchronisation in the capture apparatus**: whether every
stream shares one clock, whether the gz→ROS bridge preserves stamps under
real-time-factor throttling, and how old an observation is at the moment a filter
would consume it.

You are **not** improving the filter, the observation model, the noise model, or
the notebooks. You are **not** re-recording data. You are **not** fixing anything
in place. See §5 for what "report, don't fix" means concretely.

---

## 2. Frozen evidence — do not modify, do not re-record

Nine captures under `logs/studies/filter_notebook/`:

| capture | steps | detections used |
|---|---|---|
| `aws_aisle_east_north` | 638 | 276 |
| `aws_aisle_west_north` | 616 | 224 |
| `aws_apron_west_to_east` | 559 | 272 |
| `aws_mid_cross_east` | 379 | 94 |
| `aws_apron_diagonal_ne` | 269 | 116 |
| `aws_apron_diagonal_sw` | 269 | 103 |
| `aws_apron_corner_left` | 363 | 160 |
| `aws_apron_arc_left` | 310 | 118 |
| `aws_apron_reverse_spin` | 295 | 121 |

These are the evidence for `pp4_1_learning_r`, `pp4_5_the_r_field` and
`pp4_6_the_observation_function`, and for the two result write-ups
`RESULT_heading_diagonals_2026-08-14.md` and
`RESULT_learning_r_after_the_bias_fix_2026-08-14.md`. Numbers quoted in those
files are cited in the thesis defence.

**Do not edit** any of:

- `logs/studies/filter_notebook/aws_*/**` — the captures themselves
- `experiments/filter_notebook/notebook_data.py` — loader, `ASSOC_TOL_S = 0.15`
  (odometry/truth join), `TRUTH_TOL_S = 0.05`
- `experiments/filter_notebook/notebook_model.py` — `GRID_HZ = 10.0`,
  `ASSOC_TOL_S = 0.06` (detection→grid-step join), class `Sequence`
- `experiments/filter_notebook/capture_aws_notebook_dataset.sh` — capture protocol
- any `pp4_*.ipynb` / `pp4_*.py`

Changing a tolerance or a grid constant silently changes every number in three
notebooks with no error and no warning. That is the failure mode this brief is
written to prevent.

**Do not launch a simulator.** `pgrep -f gazebo` / `pgrep -f gz sim` before any
process you start, and if you believe you need a run, stop and ask instead.

---

## 3. Already settled — do not re-derive

### 3.1 The stamp chain is clean

The one bug that would have contaminated everything — observations stamped at
inference-*finish* instead of image-*capture* — **is not present**. It was traced
end to end through four hops on 2026-08-16: the observation's `timestamp_s` is
copied from the image header, which comes from Gazebo. Verifying this again is
wasted effort. If you want to spot-check it, spot-check it; do not re-open it as
an investigation.

### 3.2 The 10 Hz grid snap is known, is analysis-side, and its cause is now known

Detections are associated to the nearest step of a uniform 10 Hz grid
(`notebook_model.py:136`). The pooled snap displacement across all nine captures,
measured 2026-08-17:

```
median 33.0 ms   mean 32.3 ms   p95 42.0 ms   (n = 1484)
→ 0.49 cm median, 0.51 cm RMS along-track at 0.15 m/s
```

**This is not a simulator defect.** Two measured facts pin it down:

1. Within any single capture, median == p95 — the snap is one *constant* value
   (9 ms to 42 ms depending on the capture), not jitter. A constant time offset at
   constant speed is a constant along-track displacement, not noise.
2. Detection stamps land **exactly** on multiples of 0.1 s (`stamp mod 0.1` takes
   the values {0.0, 0.1} only; detection period is 0.2 s, i.e. 5 Hz). Odometry
   runs at 50 Hz.

So the whole snap comes from the *phase of the grid origin*: `Sequence` starts its
grid at `lo = route_window[0]`, an arbitrary time, while the data sits on exact
0.1 s boundaries. Quantising `lo` to a multiple of `1/grid_hz` would drive the
snap to exactly zero for every capture.

That one-line change is **not yours to make** — it shifts every grid step and
re-runs three notebooks. Note it, do not apply it.

Corollary you may find useful: the snap can explain at most ~0.6 cm of the
~4 cm heading-dependent offset term. It is not that term's explanation.

### 3.3 Blur

Zero by construction. Nothing to find. Do not spend time here.

---

## 4. The four questions to answer

1. **One clock.** Does every node contributing to a capture run with
   `use_sim_time: true`, and does every stream carry sim time rather than wall
   time? Start from `src/experiments/launch/warehouse_aws_notebook_capture.launch.py`
   and follow to every node it brings up (camera bridge, detector, odometry,
   ground-truth bridge, the recorder).

2. **Stamp preservation under RTF throttling.** Captures were recorded at
   `RTF=0.5` (`capture_aws_notebook_dataset.sh:155`, set via an ignition
   `Physics` service request). Does the gz→ROS bridge preserve image header
   stamps under throttling, or does any hop restamp with arrival time? Any node
   that works in wall time has its effective latency changed by a factor of two
   at RTF 0.5 — that is the specific thing to look for.

3. **Consistency of `/odom_noisy` and `/ground_truth_tf` with the images.** One
   concrete lead, already located: the ground-truth bridge publishes transforms
   with a **zero header stamp**, and the recorder substitutes the node clock —
   see `experiments/filter_notebook/record_demonstration_capture.py:126-142`.
   That substitution is deliberate (without it there is no truth at all), but it
   means truth rows are stamped at *receive* time, not at the pose's own time.
   Quantify that gap. Truth is evaluation-only, so an error here biases scoring,
   not filtering — say which of the two any finding affects.

4. **End-to-end observation age.** At the moment a filter would consume a
   detection, how old is the underlying image, in sim seconds? Measure it, don't
   estimate it. Report median and tail, per capture.

---

## 5. Deliverable

A single markdown report at
`logs/studies/filter_notebook/RESULT_synchronisation_audit_2026-08-17.md`.

Rules:

- **Report findings; do not fix them.** For anything you would change, write the
  file, the line, the proposed diff, and — explicitly — which notebooks and which
  published numbers would have to be re-run if it were applied. Then stop.
- **Measure, don't reason.** Every number in the report must come from a command
  you ran, with the command shown. "Should be" and "presumably" do not belong in
  it. If you cannot measure something without a new capture, say so and leave it
  unmeasured.
- **Separate the three severities**: (a) contaminates existing results,
  (b) affects future captures only, (c) cosmetic. Nothing in §3 is (a) — it has
  already been checked.
- New scripts go in `experiments/filter_notebook/` with an `audit_` prefix, and
  must not import-time mutate anything. Read-only against the captures.

---

## 6. Explicitly out of scope

- **Speed dependence.** Every drive ran at 0.15 m/s. This is a real gap, it is
  genuinely untested, and it is being handled in the main session as three new
  captures on the existing apparatus. Do not capture, and do not speculate about
  it in the report.
- Any change to the filter, the R model, the offset model, or the visibility /
  EFE cost terms. The visibility term in particular is frozen method and is not
  tunable as a fix for anything.
- Re-deriving the bias fix, the R_miss result, or the offset decomposition.

---

## 7. Reproducing the snap measurement

From `experiments/filter_notebook/`:

```python
import numpy as np, notebook_data as nd, notebook_model as nm
tag = "aws_aisle_east_north"
cap = nd.load_capture(tag)
seq = nm.Sequence(cap, nd.load_truth(tag), window=nd.route_window(tag))
err = [abs(seq.stamps[int(np.argmin(np.abs(seq.stamps - d.stamp)))] - d.stamp)
       for c in seq.cameras for d in cap.detections[c]]
err = np.array([e for e in err if e <= nm.ASSOC_TOL_S])
print(np.median(err), np.percentile(err, 95), len(err))
```
