# Workstream map

This directory splits the thesis programme into bounded assignments that can be handed to
separate chats without giving each chat ownership of the whole repository. It is a
coordination layer, not a second status system: `research/registry.yaml` remains the only
machine-readable authority and `research/STATUS.md` remains generated.

## Integration rule

The integration chat is the only owner of:

- `research/registry.yaml` and generated `research/STATUS.md`;
- `research/README.md`, immutable `research/01_questions.md`, and
  `research/09_decisions_and_risks.md`;
- cross-workstream source changes, status transitions, campaign activation, commits, and
  evidence promotion approval.

Specialist chats edit only the paths listed in their handoff. If a registry or shared-file
change is needed, they describe it in their final report; the integration chat applies it.
This avoids simultaneous chats silently forking the research story or overwriting one
another.

## Workstreams

| ID | Assignment | Owns | Depends on | Can run now? |
|---|---|---|---|---|
| WS01 | Claims and method taxonomy | claim prose and estimator-family boundaries | immutable SQ1-SQ4 and registry C1-C6 | Yes |
| WS02 | Assumptions and controls | assumptions, world/camera design, noise and frozen controls | WS01 claim boundaries | Yes |
| WS03 | Validation, evidence and figures | reviewer attacks, validation matrix, figure contract and promotion manifest | existing evidence; WS01 claim IDs | Yes |
| WS04 | Pixel-to-ground measurement path | the three current dirty source/test paths and `experiments/pixel_ground_path/` | WS02 measurement assumptions | Yes, but no runtime promotion yet |
| WS05 | Correlated-error closed-loop protocol | current paper scope and `experiments/closed_loop_calibration/` | WS01-WS04 | Design only until blockers close |
| WS06 | Campaign setup and readiness | one pre-campaign entry point and its tests/report | frozen WS05 arms and metrics | Not yet |
| WS07 | Reliability-source benchmark | `experiments/usable_observation/` and source-comparison paper scope | WS01, WS02 and WS04 interface decision | Design only; after current paper for execution |
| WS08 | Camera management | `experiments/multicamera_fusion_extension/` | winning frozen WS07 fields | No |

## Scientific dependency

```text
immutable SQ1-SQ4 and C1-C6
             |
      WS01       WS02       WS03
        \          |          /
         +---------+---------+
                   |
                 WS05 <----- WS04
                   |
                 WS06
                   |
           closed-loop campaign

WS01 + WS02 + WS04 ---> WS07 ---> WS08
```

WS01-WS03 can run in parallel. WS04 may finish its isolated analysis, but neither WS05 nor
WS07 may adopt its runtime interface until WS02 accepts the measurement assumptions. WS06
cannot freeze checks until WS05 selects scientifically valid campaign arms. WS08 remains
downstream because refitting a quality field for each camera
policy would confound estimation with management.

## Immediate chat wave

Start three chats now, using the complete prompts in their handoffs:

1. `WS01_scientific_contract/HANDOFF.md`
2. `WS02_assumptions_controls/HANDOFF.md`
3. `WS03_validation_evidence/HANDOFF.md`
4. `WS04_pixel_ground/HANDOFF.md`

After their reports are integrated, start WS05. Start WS06 only after WS05 has frozen the
arms, metrics, seeds, and analysis contract. WS07 can be discussed in parallel as a design
chapter, but it must not consume Gazebo campaign time before the current paper gate closes.

## Common completion contract

Every specialist report must contain:

- files changed and files deliberately not changed;
- decisions made, unresolved questions, and assumptions relied on;
- exact evidence inspected, including paths and hashes where relevant;
- tests/checks run and their results;
- proposed registry changes, without editing the registry;
- a short handback stating which downstream workstream is now unblocked.

No chat may delete or archive data, launch a confirmatory campaign, change the active
research focus, or promote a claim/figure to `LOCKED` without integration review.
