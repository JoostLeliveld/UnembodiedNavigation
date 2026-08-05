# Paused and archived research

This folder is the front door for work deliberately removed from the active research
surface on 2026-08-05. It contains indexes, not bulk payloads.

The recoverable payload is stored outside the active Git repository at:

```text
/home/joostleliveld/Thesis/_archive/UnembodiedNavigation_paused_2026-08-05/
```

Nothing in the archive is evidence-approved merely because it exists. Restore a path only
when the active research question explicitly needs it, then re-check its evidence class,
world, detector, calibration, and runtime contract.

- [`ACTIVE_RESEARCH_SET.md`](ACTIVE_RESEARCH_SET.md) defines what remains active.
- [`ARCHIVE_CATALOG_2026-08-05.md`](ARCHIVE_CATALOG_2026-08-05.md) lists the complete
  archive/cleanup classification.
- [`ARCHIVE_SNAPSHOT_2026-08-05.md`](ARCHIVE_SNAPSHOT_2026-08-05.md) records payload
  sizes, file counts, retained critical paths, and the post-move test result.

## Rules

1. The active paper is the correlated-bias → honest-belief → closed-loop consequence
   workstream.
2. Canonical evidence is never deleted. Bulk raw data may move to cold storage, but its
   manifest and source result remain.
3. A paused idea does not re-enter the active tree without a one-sentence research
   question, a discriminating experiment, and a stop rule.
4. New smoke runs and failed captures go directly to the external archive after diagnosis.
5. Do not add a new world, detector successor, GP variant, or fusion rule before the
   closed-loop campaign is frozen and analysed.
