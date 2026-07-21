# Option A — commissioning external-camera trust maps under uncertain robot poses

Investigation study (8 experiments, RQ1–RQ6). **This folder is the reference layout for
new studies** (see `/CLAUDE.md`): analysis scripts + shared module here, all outputs in
`logs/studies/optionA_commissioning/<expN_name>/` (figures + `RESULTS.md` per experiment,
master `INDEX.md` + `SHOWCASE.png` at the top).

Run: `python3 expN_*.py` from this directory (each script is standalone; ~1–10 min each).

| file | what |
|---|---|
| `optA_common.py` | study-shared code: paths, plot style, event loading; delegates GP math to `scripts/visibility_comparison/fit_belief_aware_gp.py` and scoring to `scripts/shared/metrics.py` |
| `exp0_confidence_audit.py` | RQ1 gate: is YOLO confidence a usable quality proxy? |
| `exp1_synthetic_gp.py` | RQ2 math: uncertain-input GP on a known synthetic field |
| `exp2_operational_mapping.py` | RQ2: point vs belief-aware on real events + degradation sweep |
| `exp34_init_budget.py` | RQ3: priors I0–I3 × commissioning data budget |
| `exp5_trajectory_smoothing.py` | RQ4: offline smoothing with camera anchors, L0–L3 maps |
| `exp6_stress_test.py` | RQ5: historical-map reuse under change, inflation vs replace |
| `exp7_planner_replay.py` | RQ6: τ→R_plan→predicted belief through the planner seam |
| `make_showcase.py` | assembles `SHOWCASE.png` from the per-experiment figures |
| `REUSE_MAP.md` | audit of which pre-existing repo assets were reused / moved / retired |

Data: pre-existing `honest_campaign_v1` + `whitenoise_campaign_v1` logs; GT/CAD are
evaluation-only throughout. Results summary: `logs/studies/optionA_commissioning/INDEX.md`.
