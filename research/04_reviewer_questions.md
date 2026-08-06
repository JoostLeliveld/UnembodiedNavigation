# Reviewer attack register

Unanswered items remain `OPEN` or `LIMITATION`; they are never removed from the narrative.

| ID | Reviewer concern | Why reasonable | Claim | Experiment / figure | Status | If unanswered |
|---|---|---|---|---|---|---|
| RQ01 | Why not use distance only? | It is cheap and often correlated with quality. | C2 | EXP-USABLE / F07 | OPEN | Limit novelty to taxonomy and deployment evidence. |
| RQ02 | Why use a GP? | Learning needs commissioning data. | C2 | EXP-USABLE / F07,F09 | OPEN | Treat GP as a null-result ablation. |
| RQ03 | Is depth realistic? | Perfect maps can leak simulator geometry. | C2,C6 | EXP-USABLE / F07,F09 | OPEN | Restrict the result to mapped geometry. |
| RQ04 | Does geometry become stale? | Warehouse layouts change. | C6 | EXP-DRIFT, EXP-USABLE / F04,F10 | OPEN | State static-layout scope. |
| RQ05 | Does the GP transfer? | Unsupported regions and new layouts cause shift. | C2,C6 | EXP-USABLE / F07,F10 | OPEN | Require recommissioning; no transfer claim. |
| RQ06 | Why is hybrid worth commissioning? | Added complexity needs measurable benefit. | C2,C6 | EXP-USABLE / F09 | OPEN | Do not promote hybrid. |
| RQ07 | What if calibration drifts? | Fixed cameras can move. | C1,C6 | EXP-DRIFT / F04 | ANSWERED | Controlled-injection scope remains. |
| RQ08 | Does this generalize beyond YOLO? | Reliability may be detector-specific. | C6 | None | LIMITATION | Claim a frozen-detector contract only. |
| RQ09 | Are cameras genuinely diverse? | Identical optics limit generalization. | C6 | EXP-COMMISSION / F01 | LIMITATION | Claim geometry and bias diversity only. |
| RQ10 | Are worlds representative? | A/B/C labels do not establish coverage. | C6 | EXP-USABLE / F09 | OPEN | Report measured properties and narrow scope. |
| RQ11 | Is fusion overconfident? | Camera errors are correlated. | C1,C5 | EXP-BELIEF / F02 | ANSWERED | Never claim independent fusion. |
| RQ12 | How expensive is commissioning? | Deployment cost may dominate accuracy. | C2,C6 | EXP-COMMISSION, EXP-USABLE / F09 | OPEN | Report sample curve and setup time. |
| RQ13 | Does prediction change navigation? | Offline score gains may be irrelevant. | C3,C4 | EXP-CL-CAL, EXP-USABLE / F06,F08 | OPEN | Keep navigation as an open hypothesis. |
| RQ14 | What makes each method fail? | Average performance hides unsafe regimes. | C2,C6 | EXP-USABLE / F10 | OPEN | No method is promoted without a failure case. |
