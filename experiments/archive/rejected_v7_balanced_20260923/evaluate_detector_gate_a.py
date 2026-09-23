#!/usr/bin/env python3
"""Gate A of the 2026-09-23 amendment: does the retrained detector replace the v5 one?

Both detectors are run through the same inference and the same frozen sensor gate on
the v7 working roles. Only D_dev enters the decision. The retrained detector passes when

* it admits more camera C opportunities on D_dev than the v5 detector, and
* its D_dev recall on reference positives is at most one percentage point lower.

False detections on reference negatives are reported but do not decide.

    python3 experiments/thesis_pipeline_lock/evaluate_detector_gate_a.py \
      --baseline GATE_DIR_V5 --candidate GATE_DIR_NEW --output report.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def summarize(gate_dir: Path) -> dict:
    admitted = collections.Counter()
    opportunities = collections.Counter()
    tp = fn = fp = negatives = 0
    with (gate_dir / "opportunities.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if r["stratum"] != "D_dev":
                continue
            cam = r["camera_id"]
            opportunities[cam] += 1
            if r["opportunity_outcome"] == "admitted":
                admitted[cam] += 1
            ref = r.get("offline_reference_class")
            if ref == "positive":
                if r["detected"]:
                    tp += 1
                else:
                    fn += 1
            elif ref == "negative":
                negatives += 1
                fp += bool(r["detected"])
    return {
        "admitted_by_camera": dict(sorted(admitted.items())),
        "opportunities_by_camera": dict(sorted(opportunities.items())),
        "recall": tp / max(tp + fn, 1),
        "true_positive": tp, "false_negative": fn,
        "false_positive": fp, "reference_negatives": negatives,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base, cand = summarize(args.baseline), summarize(args.candidate)
    c_gain = cand["admitted_by_camera"].get("camera_C", 0) - base["admitted_by_camera"].get("camera_C", 0)
    recall_drop_pp = 100.0 * (base["recall"] - cand["recall"])
    passed = c_gain > 0 and recall_drop_pp <= 1.0
    report = {
        "schema": "thesis_detector_gate_a.v1",
        "population": "D_dev only",
        "baseline": base, "candidate": cand,
        "camera_C_admitted_gain": c_gain,
        "recall_drop_pp": recall_drop_pp,
        "passed": passed,
        "decision": "retrained detector replaces v5" if passed else "keep v5 detector",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("camera_C_admitted_gain", "recall_drop_pp",
                                             "passed", "decision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
