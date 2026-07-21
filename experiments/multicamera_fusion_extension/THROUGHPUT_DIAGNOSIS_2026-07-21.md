# Four-camera throughput: why not 3 Hz, and does it actually block the paper?

Investigation date 2026-07-21. Evidence: `runtime_pilot_20260717_retry2/runtime_readiness.json`,
`LAPTOP_RUNTIME_PROBE_2026-07-20.md`, `preflight_host_gpu_20260720.json`.
Host: **Quadro P2000, 4 GiB VRAM**; system RAM 5.8 GiB free with swap ~100% used.

## The measured facts

| camera | sim_hz | wall_hz | inference p50 (ms) | fresh frac |
|---|---:|---:|---:|---:|
| A | 2.64 | 0.85 | **986.6** | 0.80 |
| B | 5.00 | 1.54 | 93.1 | 1.00 |
| C | 5.00 | 1.54 | 110.2 | 1.00 |
| D | 5.00 | 1.54 | 103.4 | 1.00 |

Simulation `real_time_factor = 0.315`. The 3 Hz gate is a **wall-clock** rate.

## Two independent causes (do not conflate them)

1. **Sim runs at ~0.31× real-time.** Cameras B/C/D render at the requested 5 Hz
   in *sim* time, but 5 Hz × 0.315 ≈ **1.54 Hz wall** — which is the accepted
   output rate. The ceiling is Gazebo rendering 4 camera sensors + physics + 40
   contact sensors (60 Hz each) on a 4 GiB GPU, NOT the detector: B/C/D infer in
   ~100 ms, which alone would support ~10 Hz. To reach 3 Hz *wall* you need
   RTF ≈ 0.6 (double the sim speed) or a 10 Hz sim cadence — neither is cheap on
   this GPU. Prior probes already showed 640×360 (4× fewer pixels) did not move
   the accepted rate, and disabling all 40 contact-sensor bridges lifted it only
   ~1.0 → ~1.15 Hz. **There is no cheap wall-clock fix to 3 Hz on this host.**

2. **camera_A is broken, independently of the rate gate.** ~986 ms inference
   (10× B/C/D) starves it below the frame rate (sim_hz falls to 2.64, fresh
   fraction 0.80), and in the captured smoke run camera_A produced **0
   detections**. This is the GIL/executor-contention pattern recorded in memory
   ([[project_detector_runtime_latency_contention]]): camera_A is on the
   separate per-camera fallback detector process (launch line 342 "Separate-
   process fallback device for camera_A"), likely split out because 4 batched
   detectors OOM the 4 GiB VRAM. So A pays a contention/queueing cost B/C/D avoid.

## The reframe that unblocks the paper

**Wall-clock rate only matters for LIVE closed-loop navigation (E8).** For
offline data collection (record once → replay every method: E3–E7, the bulk of
the extension), the recorded observations carry **sim timestamps at 5 Hz** for
B/C/D — fully dense. A 0.31× RTF just means a 60 s route takes ~190 s of wall
time to record. The offline replay/evaluation apparatus
(`tools/replay_sweeps.py`, `tools/experiment_evaluators.py`) uses sim time, so it
does not care about wall Hz at all.

**Conclusion:** proceed at the achievable rate now. 3 Hz wall is not required to
collect E3–E7 data; it is only required to run E8 live, which is deferred.

## Ranked fix candidates (for "later")

1. **Fix camera_A first (highest value, rate-independent).** A dead 4th camera
   corrupts the multi-camera story more than slow wall time does. Options, in
   order of effort: (a) give A the same batched fast path as B/C/D if VRAM
   allows a 4-image batch after freeing memory; (b) TorchScript / single-thread
   executor / move inference off the spin thread (the memory-noted GIL fix, which
   the other workstream's torchscript successor is attempting); (c) run A's
   detector on CPU on a dedicated core if the GPU can't hold a 4th model.
2. **Raise RTF for data collection.** Disable the 40 contact-sensor bridges
   during *recording* (they are evaluation-only geometry; re-enable for
   collision-valid runs), drop physics/`max_step_size`, and cut camera FOV/res —
   target RTF ≥ 0.6. Prior evidence says each lever alone is weak; combined they
   may help.
3. **Reduce concurrent camera count** (e.g. two batched pairs, or stagger),
   trading coverage for rate — only if live E8 is required.

Owner note: the perception nodes / launch / detector configs are the parallel
commissioning workstream's active (uncommitted) files. Any camera_A fix should
be coordinated with them, not applied blindly here.
