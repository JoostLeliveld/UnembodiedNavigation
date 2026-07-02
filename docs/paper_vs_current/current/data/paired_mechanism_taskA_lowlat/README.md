# paired_mechanism_taskA_lowlat source data

This directory contains the exact run files used to generate `paired_mechanism_taskA_lowlat.pdf`.

Source campaign: `logs/visibility_comparison/honest_campaign_v1`

Task: `route_apron_to_a3_mid`, seed: `0`

Regenerate the figure with:

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
PAIRED_CAMP=honest_campaign_v1 PAIRED_TASK=route_apron_to_a3_mid PAIRED_SEED=0 \
  python3 scripts/paper_figures/make_paired_mechanism.py
```

The selected runs use the locked keep-in warning-band no-go setting, YOLO masks disabled, and the warehouse visibility GP artifact. The full runtime values are recorded in each condition's `run_manifest.json`.
