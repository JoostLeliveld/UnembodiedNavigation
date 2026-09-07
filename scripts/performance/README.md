# Runtime performance checks

The registered localization evidence remains tied to its exact selection and hashes. These
tools provide faster development feedback and prospective runtime measurements; they do not
create or replace navigation evidence.

## Low CPU simulator profile

`warehouse_v2_low_cpu.world.sdf` retains the `warehouse_v2` geometry, camera models, lighting,
collision shapes and calibration profile. It changes only:

- ODE physics: 1000 Hz / 1 ms to 200 Hz / 5 ms.
- 46 contact sensors: 60 Hz to 20 Hz.

At the configured maximum speed of 0.22 m/s, those periods correspond to 1.1 mm of travel per
physics step and 1.1 cm between contact samples. The dedicated campaign config is
`experiments/icra_commissioning/network_navigation_runtime_fast.yaml`; keep its outputs in a
separate development ledger.

After verifying CUDA, use `network_navigation_runtime_fast_gpu.yaml` to combine this simulator
profile with detector device 0. The CPU version remains available as a fallback.

A 12-second headless A/B probe on 2026-09-07 measured 0.4268 simulated seconds per wall second
for the registered world and 0.6988 for the low CPU world. This is a 1.64x simulator speedup,
or about 39% less wall time for the simulator-bound portion. It is not a full navigation run.

Dry-run the launch expansion without starting ROS or Gazebo:

```bash
python3 scripts/visibility_comparison/run_visibility_campaign.py \
  --config experiments/icra_commissioning/network_navigation_runtime_fast.yaml \
  --log-root logs/performance/low_cpu_dry_run --dry-run
```

## Focused checks

```bash
python3 scripts/performance/check.py --suite performance
```

Pass `--selection` only with a selection JSON explicitly registered in
`docs/localization_metrics_registry.json`. The checker verifies every frozen file hash and uses
the repository's canonical alignment/event validators.

## Component benchmarks

```bash
python3 scripts/performance/benchmark_planner.py --out logs/performance/planner_probe
python3 scripts/performance/benchmark_detector.py --devices cpu \
  --out logs/performance/detector_probe
```

The planner cache is content-addressed and invalidates on numerical inputs, relevant source,
CasADi/NumPy/platform identity and compilation settings. `--planner-jit` remains opt-in because
the measured benefit was small. The detector benchmark uses exact decoded-image hashes from the
captured dataset and records the source checkpoint hash.

The CUDA failure was traced below PyTorch: `/dev/nvidia-uvm` returned `EIO`, after kernel events
Xid 31 (MMU fault) and Xid 154 (`Node Reboot Required`). Driver module and user libraries both
reported 580.173.02. Reloading the unused `nvidia_uvm` module restored direct driver initialization
and a real PyTorch allocation without a reboot.

An isolated five-repeat benchmark of the commissioned 960 px checkpoint measured 1,004.5 ms CPU
versus 99.0 ms GPU median per five-camera batch, a 10.15x speedup. GPU p95 was 109.2 ms. Across 15
exact hashed images, CPU/GPU comparison changed zero hit decisions; maximum box and confidence
deltas were 0.000122 px and 4.77e-7, within the declared 0.02 px and 1e-4 tolerances.

Verify the host context and rerun the benchmark with:

```bash
python3 scripts/performance/check_cuda.py
python3 scripts/performance/benchmark_detector.py --devices cpu 0 \
  --out logs/performance/detector_cpu_gpu_after_restart
```

Do not lower the inference resolution in an evidence run without a separate detector and
localization parity study: the checkpoint was trained at 960 px.
