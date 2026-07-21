# NN — <chapter title, phrased as the investigation>

<!-- Chapter README template — investigative format. Chapters are MANIFESTS: claims, gates,
     pointers. Code goes in experiments/<study>/, outputs in logs/studies/<study>/, locked
     artifacts in paper_artifacts/. Media views (figures/, videos/) are built by
     ../_tools/link_media.py — add entries there, don't copy files. -->

**Question.** <the falsifiable research question this chapter answers>

**Status:** LOCKED | ACTIVE | PARTIAL | PLUMBING | PLANNED | FUTURE.
**World:** aws_1cam / full_4cam (warehouse_full_4cam.world.sdf) / synthetic.

## What a contribution here looks like

<the claim statement, verbatim as it would appear in the thesis, plus what must be true to
earn it — including the honest null/negative outcome if that is also publishable>

## The results we're aiming for

<the concrete figures/tables with their decision criteria — "Fig NNx: aim = ..." — mark ONE
as the decision figure. Include forbidden assumptions here (GT/CAD as deployment input,
perfect poses, oracle maps).>

## Implemented now

| Item | Tag | Note |
|---|---|---|
| <artifact/code/finding> | established / measured_in_sim / model_plumbing / open | <honest caveat> |

## Gap → next experiment

<the smallest experiment that moves the chapter; name the study folder it goes in>

## Gate

<the go/no-go rule: what result kills, demotes, or freezes this chapter; exact commands and
frozen configs referenced here>

Update [evidence.yaml](evidence.yaml) and `../registry.yaml` when status changes.
