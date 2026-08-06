# Decisions, risks, and archive index

## Locked decisions

- `UnembodiedNavigation` is the only live research repository.
- The current publication is the correlated-error and belief-honesty paper.
- The reliability-source comparison is the next thesis chapter.
- Camera management follows source estimation and uses frozen reliability fields.
- Evidence is Gazebo-only, one robot, and 2-D position; no hardware claim is made.
- Null results, summaries, manifests, provenance, and decisive figures are permanent.
- Generated data is disposable; irreplaceable raw data moves only after hash verification.

## Recovery

The complete pre-consolidation tracked state is available at Git tag
`pre-research-consolidation-2026-08-06`. Removed tracked trees should be recovered from that
tag, not replaced with pointer READMEs.

Cold-storage manifests are stored both beside the archived content and under
`/home/joostleliveld/Thesis_Cold_Archive/manifests/`. Each records source, destination,
size, file count, SHA-256 manifest, reason, evidence references, date, and restore command.

## Verified cold-storage transfers — 2026-08-06

| Source | Bytes | Entries | Payload-manifest SHA-256 |
|---|---:|---:|---|
| `_archive` | 15,448,978,927 | 71,319 | `47c5cdb2ad24c9d028de89737ebd36e8551b73d126ab6b6eb5647f022de9dce9` |
| `RobotControlExternalCamera` | 330,666,321 | 2,907 | `4436841489019b15f5eacc1e2d215a0c31486e32dfac277ef2e005ed268e4a74` |
| `thesis-report` | 94,953,537 | 857 | `7589636b8b5ced5e6e74bb090c8edbbabb550fed62442b3ecc0f2096284ef1d6` |
| `midterm_presentation` | 214,601,653 | 399 | `5e28786f31f940ad710dc3811542ac52de80e2d2c702642fc37364c1dec91490` |
| `side_projects` | 410,679,425 | 5,455 | `a9a8917d7dabafc603a4d21e8e39e0725771a3cd410b4a94635c14dc64cb0f88` |
| `meeting_results_update_2026-07-27` | 1,374,621 | 5 | `26f699a183953da7c70f6b5e85ed564275a34789fb0ec559035cb3e36babb281` |
| `_private_ai_notes` | 318,267 | 34 | `279a9dc9f60953d365a70585d53e2ff9c481b3579bc736659d67fe8c19e2ed19` |
| `.claude` | 311 | 1 | `7f48210dce3e3f81b174d8d007e1684d51264f63e72fbd0f04f54d85d47bf6f2` |
| `REPO_ORGANIZATION_AUDIT_2026-07-15.md` | 13,017 | 1 | `d7541811daa7bf390efe6afe61332a38e59145b24d2de5a72a8c98948579963a` |

All destinations are under
`/home/joostleliveld/Thesis_Cold_Archive/workspace_repositories/`. The source was removed
only after the relative path, type, size, and SHA-256 inventories matched. Per-entry hash
inventories and restore commands live beside each archived payload and in the central
manifest directory.

## Consolidation verification — 2026-08-06

- Recovery state: four coherent preservation commits and tag
  `pre-research-consolidation-2026-08-06`.
- Registry: validator, generated status, experiment metadata synchronization, and hygiene
  checks pass.
- Repository tests: `912 passed`; three non-failing dependency/Matplotlib warnings.
- Clean checkout: detached worktree at `7f4ce18`; all seven colcon packages rebuilt with
  `--symlink-install` in 1 minute 6 seconds.
- Campaign smoke: the v3 `mc_central_ns` arm launched from the clean build; Gazebo, four
  camera bridges, batched detector, runtime-contract publication, camera manager, logger,
  and planner initialized. The robot executed repeated valid CasADi plans until the
  intentional 55-second timeout.
- Generated output: old `build/`, `install/`, and `log/` were removed only after the clean
  build and launch passed; these paths remain explicitly ignored and disposable.
- Remaining dirty files after consolidation belong to the independently preserved
  pixel-to-ground projection work and were not included in cleanup commits.

## Principal risks

| Risk | Control | Residual limitation |
|---|---|---|
| Cleanup destroys current work | Four coherent commits plus recovery tag precede removal. | Untracked external files require separate manifests. |
| Research state forks again | Registry is machine authority; STATUS is generated. | Narrative prose can still become stale and must not carry status. |
| Oracle leakage | Operational/evaluation input split is validated. | Runtime code review remains necessary. |
| Campaign sprawl | One active experiment and offline promotion gates. | Exploratory work must remain outside active studies. |
| Cold archive becomes a junk drawer | Hash manifests and one archive index. | Storage health/backups are outside repository control. |
| Four cameras are overgeneralized | Report geometric and bias diversity only. | Optical/hardware diversity is not tested. |
| Better prediction does not help navigation | Treat C4 as open and preserve a null. | A null narrows the contribution to representation/estimation. |

## Campaign hold — 2026-08-06

`EXP-CL-CAL` remains the active scientific focus, but it is in protocol-resolution work, not
campaign execution. Separate audits found four blockers:

- the active v2-v3 arm definition predates held-out evidence that explicitly supersedes both
  artifacts with the minimal v4 pipeline;
- generated `_clv2.yaml` and `_clv3.yaml` currently contain seed 0 only, not the documented
  seeds 0-4;
- the analyzer lacks the preregistered NEES/NIS, correction acceptance/age, uncertainty
  intervals, plot output, and complete-pair gate;
- the current world changed after the July GP fields, whose manifests do not bind a world
  hash.

No confirmatory config generation, readiness run, or campaign is permitted until WS05
resolves the scientific protocol and WS06 passes the resulting fail-closed readiness gate.
The workstream ownership and paste-ready separate-chat handoffs live under
`research/workstreams/`.

## Artifact hashes

Detector, calibration, GP, and campaign hashes are canonical entries in `registry.yaml` and
are checked by the registry/hygiene tooling before campaigns and archive moves.
