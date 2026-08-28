# If the reading has to be unbiased, what happens to the rest?

**Serves** PLAN.md sentences 2 and 4. Outputs in `logs/studies/unbiased_observation/`.

## The question

The hull reading is unbiased whenever the detector's box covers the whole robot (+0.2 cm),
and leans outward by up to 12.5 cm when an obstruction cuts the box short. So an unbiased
detector is not something to build — it is something to **choose**, by refusing the cut
sightings. This study asks what that choice costs and what it changes.

The interesting part is that the cost lands somewhere useful:

> A refused sighting is an **absent** measurement, not a wrong one.
> Tightening the gate moves error out of `R_c(s)` and into `p_c(s)`.

That is the paper's own thesis applied to its own sensor. If the two quantities really are
separable, then trading one for the other should be a *choice with a visible price*, and this
study is where that price is measured.

## What is NOT done here

No synthetic readings, no simulated detector, no injected noise. Every rule on the ladder is
computed from a detected box and a predicted one on the real commissioning capture, so every
row is a rule the runtime could actually run tomorrow. "An unbiased detector" means a
stricter admission check, nothing else.

## Files

| file | the one question it answers |
|---|---|
| `gate_ladder.py` | what does each admission rule cost in availability, and what does it buy in bias? |

    python3 experiments/unbiased_observation/gate_ladder.py

Offline, about a minute, on `logs/perception_datasets/warehouse_v2_yolo_shared_20260822`
(4,797 detected sightings over 11,585 attempted, 382 floor positions, five cameras).

## What it measures, and why each column is there

- **kept** — share of *attempted* sightings that become measurements. The denominator is
  every trial, detected or not, so tightening the gate cannot hide its own cost.
- **lean / worst camera** — the systematic outward error along the line of sight, pooled and
  for the worst single camera. This is the quantity being bought.
- **dead places / one camera or none** — the shape of the availability field. A field that is
  the same everywhere gives a route planner nothing to choose between, so how *uneven* it is
  decides whether the planning experiment can exist at all.

## Still owed

A closed-loop drive with the strict gate in force. Everything here is measured on a parked
robot at known places; what it cannot show is what the belief does when the corrections stop
arriving, which is the half that matters for the route claim.
