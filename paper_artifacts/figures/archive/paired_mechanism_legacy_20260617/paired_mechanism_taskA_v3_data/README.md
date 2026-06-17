# paired_mechanism_taskA source data

This directory contains the exact run files used to generate `paired_mechanism_taskA.pdf`.

Source campaign: `logs/visibility_comparison/aws_f31b1_v3_figdata`

Task: `F31_b1_apron_a3_mid`, seed: `0`

Regenerate the figure with:

```bash
cd /home/joostleliveld/Thesis/UnembodiedNavigation
PAIRED_CAMP=aws_f31b1_v3_figdata PAIRED_TASK=F31_b1_apron_a3_mid PAIRED_SEED=0 \
  python3 scripts/paper_figures/make_paired_mechanism.py
```

The selected runs use the locked keep-in warning-band no-go setting, YOLO masks disabled, and the aws_gp_v7b reliability artifact. The full runtime values are recorded in each condition's `run_manifest.json`.
