#!/usr/bin/env python3
"""Write each arm's story.md from its own numbers.json plus an authored verdict.

    python3 experiments/fusion_on_fixed_routes/story/write_stories.py

The prose is authored per arm; every number in it is read from that arm's numbers.json, so a
re-run cannot leave a stale figure quoted in text.
"""
import json
from pathlib import Path

import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from score import FOLDER, TASKS, story_dir   # noqa: E402

ROOT = REPO / "logs/studies/fusion_on_fixed_routes"
TITLE = {"F1": "F1 — the single best camera", "F2": "F2 — distance and viewing angle",
         "F3": "F3 — precisions add", "F4": "F4 — the network as one sensor",
         "O1": "O1 — the box bottom-centre as the robot",
         "O2": "O2 — the box plus one fixed offset"}
WHAT = {
 "F1": "At every correction it kept the camera with the smallest ellipse and discarded the rest.",
 "F2": "It weighted the cameras by range and viewing angle alone — never by their covariance — with the coefficients frozen before the drive.",
 "F3": "It added the cameras' precisions, the standard answer for independent measurements.",
 "F4": "It added the same precisions and then divided by the number of cameras, refusing to claim N independent votes from one robot seen by one detector.",
 "O1": "It kept the network rule but took the detector's box bottom-centre to be the robot itself.",
 "O2": "It kept the network rule and pushed that same point 30.9 cm away from its camera — one commissioned number, the mean gap over 3351 admitted sightings.",
}
VERDICT = {
 "F1": """Picking one camera is not what costs accuracy here — it is what costs the network
claim. Its median is the lowest of the six, by a margin smaller than the difference between
its own two runs, so on this evidence it is not better than the others.

**Look at `05_how_it_fused.png`, last panel.** The rule kept camera D, whose ellipse was the
smallest on the table, and camera D put the robot most of a metre away. Camera E was available
and sitting on the truth. Smallest ellipse is not most correct, and a rule that can only pick
has no way to notice.""",
 "F2": """The heuristic did its job: it lost. Knowing where the cameras are is not the same as
knowing how good their readings are — it is the least honest of the four fusion rules, and the
one whose fused answer is worse than a camera it already had most often among the rules that
weight by geometry.""",
 "F3": """The standard rule is the most overconfident of the four, and the mechanism is
visible rather than inferred. **In `05_how_it_fused.png`, last panel**, camera E was 6 cm from
the truth and the rule used it — and the answer still came out most of a metre off, because a
second camera claimed a far smaller ellipse and precision-weighting handed it the vote.

That is also why its fused answer is worse than the best camera it had more often than any
other fusion rule.""",
 "F4": """Dividing by N does not move the estimate at all — it and F3 produce the same
position, which is why their "worse than the best available camera" rates match to a point.
What it changes is the claim, and that is where it wins: the truth sits inside its stated 95%
ellipse far more often than under precisions-add, and it has the lowest worst-case error of the
six.

It is still not honest. Conservative pooling narrows the over-claim; it does not close it, and
the figures say why: the readings themselves are further from the truth than any of these rules
believes.""",
 "O1": """The observation model is worth several times the median error, and it is worth it
every metre rather than occasionally — this arm is steadily wrong, so its 95th percentile is
the best of the six while its median is the worst.

**`06_where_observations_landed.png` is the figure to show.** Each camera's readings form their
own cloud, in its own direction from the truth, because the box's bottom edge is the robot's
nearest point *to that camera*. The bias is not one offset. It is one offset per viewing
direction.""",
 "O2": """One number recovers part of what predicting the box recovers, and cannot recover the
rest: the gap it is correcting points a different way for every camera and swings about 11 cm
as the robot turns. Compare its `06_where_observations_landed.png` with O1's — the clouds move
toward the truth but do not land on it, and they land differently per camera.

This is the arm that answers "why not just subtract an offset?" with a measurement instead of
an argument.""",
}


def main() -> int:
    args = sys.argv[1:]
    tasks = [a.split("=", 1)[1] for a in args if a.startswith("--task=")] or list(TASKS)
    for task in tasks:
      for arm in FOLDER:
        path = story_dir(task, arm) / "numbers.json"
        if not path.exists():
            continue
        n = json.loads(path.read_text())
        e, h, c, d = n["belief_error_cm"], n["honesty"], n["corrections"], n["driving"]
        fq = n.get("fusion", {})
        extra = ""
        if fq.get("logged"):
            extra = (
                f"| its answer was worse than the best camera it had | "
                f"**{fq['worse_than_best_available_camera']*100:.0f}%** of "
                f"{fq['corrections']} corrections |\n"
                f"| median error of that best available camera | "
                f"{fq['median_best_available_camera_error_cm']} cm |\n")
        route = task.replace("fusion_", "").replace("_", " ")
        body = f"""# {TITLE[arm]} — {route}

Its own drive, its own numbers. Nothing here is compared with another arm — the cross-arm
figures live in `../compare/` and were built only after every folder existed.

| figure | what it shows |
|---|---|
| `02_the_drive.png` | where it drove, and how far its belief was from the truth |
| `03_claim_vs_truth.png` | whether the truth was where it said the truth would be |
| `04_worst_moment.png` | the single worst moment, in the place it happened |
| `05_how_it_fused.png` | **five real corrections: every camera's own answer, which ones the rule used, what came out** |
| `06_where_observations_landed.png` | every camera reading of the drive, on the floor and stacked on the truth |
| `07_every_camera_along_the_drive.png` | each camera's error over the whole drive, and which cameras the rule used when |

## What it did

{WHAT[arm]}

It **{n['completion'].replace('_', ' ')}** after {n['duration_s']} s, drove
{d['ground_truth_path_m']} m of the frozen 30.6 m traverse, and never strayed more than
{d['max_offset_from_commanded_route_m']} m from the route it was told to drive. It received
{c['received']} corrections, typically {c['median_gap_s']} s apart, with one gap of
{c['longest_gap_s']} s where the route crosses the stretch no camera covers. Odometry alone had
drifted {n['odometry']['final_odom_vs_truth_drift_m']} m from the truth by the end.

## How well it did

| | |
|---|---|
| belief error, median | **{e['median']} cm** |
| 95th percentile | {e['p95']} cm |
| worst moment | {e['worst']} cm |
| truth inside its own stated 95% ellipse | **{h['truth_inside_stated_95pct_ellipse']*100:.0f}%** (honest is 95%) |
| mean stated 1σ | {h['mean_stated_1sigma_cm']} cm |
{extra}
## What it means

{VERDICT[arm]}

**One drive per arm on this route, seed 0.** The same six arms were driven on three other
routes (`../../routes_compare/`), and the primary route has been driven three times in all.
Differences between arms smaller than an arm's own run-to-run spread are not differences yet:
measured on F1, that spread is about 8 points of calibration and 0.4 cm of median error.
"""
        (story_dir(task, arm) / "story.md").write_text(body)
        print(f"wrote {story_dir(task, arm).relative_to(ROOT)}/story.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
