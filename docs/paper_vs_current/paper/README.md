# Paper snapshot (the "before")

Frozen artifacts from the IWAI-paper campaign, for side-by-side comparison with
the honest re-run under `../current/`. All items here are the genuine paper
baseline (paper detector `aws_yolo_simseg_v2`, paper GP `aws_gp_v7b`, the runner
that silently ran `keep_out`, odom-as-truth metrics).

```
paper/
├── aws_f31b1_final_config.yaml   the exact paper campaign config
├── figures/
│   ├── paired_mechanism_taskA_PAPER.pdf   task-A C1/C2 paired mechanism (2026-06-12,
│   │                                      from _paper_runs/paired_mechanism_clean_verify,
│   │                                      aws_gp_v7b). This is the "before" we regenerate.
│   ├── gp_pipeline_aws.png (+ caption, provenance)
│   ├── localization_pathway.png (+ caption)
│   ├── problem_setup_camera.png, problem_setup_snapshots.png
│   ├── yolo_training_clarification.png
│   └── robustness_spread.png (+ provenance)   headline C1-vs-C2 robustness figure
└── data/
    ├── paired_mechanism_taskA/    seed-0 C1 & C2 run data for the paired figure
    │   ├── campaign_log.json
    │   ├── C1/  (experiment.csv, perception.csv, global_plan.csv, run_summary.json, …)
    │   └── C2/
    └── robustness_campaign_headline/   curated outcomes of the paper robustness
                                        campaign (campaign_log.json + 40 run_summary.json;
                                        raw per-timestep experiment.csv and ros_logs
                                        intentionally excluded — 52 MB, kept in
                                        logs/visibility_comparison/_paper_runs/).
```

Sources (full raw logs stay in the repo, not duplicated here):
- Paired figure/data: `logs/visibility_comparison/_paper_runs/paired_mechanism_clean_verify`
- Robustness headline: `logs/visibility_comparison/_paper_runs/robustness_campaign_keepout_lanegraph_v1`
  (README there designates it as the paper robustness-spread + table source).

Paper headline (from that robustness campaign): C2 16/20 goal vs C1 12/20;
collisions 2 vs 8 — but see `../README.md` §3: this "C2 > C1" was largely a
`keep_out` corner-cut artifact (the runner never forwarded `nogo_mode`), and the
collisions were odom-as-truth false positives. The `current/` re-run corrects
both.
