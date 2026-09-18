# Runtime performance checks

The registered localization evidence remains tied to its exact selection and hashes. These
tools provide faster development feedback and prospective runtime measurements; they do not
create or replace navigation evidence.

## Low CPU simulator profile (RETIRED 2026-09-18)

`warehouse_v2_low_cpu.world.sdf` was removed. It was a fidelity-reduced derivative
that claimed, in its own header and in this file, to retain the `warehouse_v2`
geometry. **It did not.** It was missing the `bin_office` include
(`aws_robomaker_warehouse_TrashCanC_01` at -8.710, -7.450), so the two worlds had
genuinely different obstacle geometry while being documented as identical.

`tests/experiments/test_low_cpu_world.py` asserted byte equality after
substituting the documented rates, and it had been FAILING on exactly that
missing include. The test was removed with the world.

It also ran coarser physics: ODE 200 Hz / 5 ms against 1000 Hz / 1 ms, and contact
sensors at 20 Hz against 60 Hz. For a result about lane departure at 1-2 sigma, a
5x coarser integrator is the wrong basis, which is why the retained world is the
full-fidelity one rather than the fast one.

The campaign configs that used it
(`network_navigation_runtime_fast.yaml`, `network_navigation_runtime_fast_gpu.yaml`,
`image_nn_drive_demo.yaml`) now point at `warehouse_v2.world.sdf`. They will run
slower; the previously recorded 1.64x simulator speedup no longer applies. If a
fast development world is wanted again, derive it from `warehouse_v2.world.sdf`
with a script rather than a hand-maintained copy, so geometry cannot drift.

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
