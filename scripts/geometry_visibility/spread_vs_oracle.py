#!/usr/bin/env python3
"""[REAL EXPERIMENT — CORRECTED] Does where/how you deposit a detection (naive vs
spread-over-belief vs oracle-at-truth) change how well the reliability field is learned?

Uses the CANONICAL belief (campaign_metrics: planner_belief, self-checked) — an earlier
version of this script used the STALE `state_x/y` field, which manufactured a fake
heavy-tailed belief error (p95 1.65 m) and a bogus "spreading helps" result. With the
real belief the error is < 0.35 m (well under the 0.6 m kernel), so position handling is
irrelevant here — naive ≈ spread ≈ oracle. Finding retracted; see docs/campaign_log_metrics.md.
"""
from __future__ import annotations
import pathlib, sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import campaign_metrics as cm

REPO = pathlib.Path(__file__).resolve().parents[2]
CAMP = REPO / "logs/visibility_comparison/honest_campaign_v1"


def main():
    ev = cm.load_detections(str(CAMP))                    # canonical belief, asserts on load
    B = np.array([e["belief"] for e in ev]); T = np.array([e["truth"] for e in ev])
    D = np.array([e["detected"] for e in ev], float)
    S = np.array([e["reported_sigma_m"] for e in ev]); S = np.where(np.isfinite(S), S, 0.02)
    err = np.hypot(B[:, 0] - T[:, 0], B[:, 1] - T[:, 1])
    print(f"[REAL, canonical belief] {len(ev)} detections")
    print(f"  belief-vs-GT error: p50 {np.percentile(err,50):.3f}  p95 {np.percentile(err,95):.3f}  "
          f"max {err.max():.3f} m  ->  0.6 m kernel, {100*(err>0.6).mean():.1f}% beyond it")
    L = 0.6; rs = np.random.RandomState(0); fold = rs.randint(0, 5, len(ev))

    def cv(P, ss=None):
        br = []
        for k in range(5):
            te = np.where(fold == k)[0]; tr = np.where(fold != k)[0]
            d2 = ((T[te][:, None, :] - P[tr][None, :, :]) ** 2).sum(-1)
            ker = np.exp(-d2 / (2 * L * L)) if ss is None else np.exp(-d2 / (2 * (L*L + ss[tr][None, :]**2))) / (L*L + ss[tr][None, :]**2)
            p = np.clip((ker * D[tr][None, :]).sum(1) / (ker.sum(1) + 1e-9), 1e-4, 1 - 1e-4)
            br.append(np.mean((p - D[te]) ** 2))
        return float(np.mean(br))

    print(f"  held-out Brier: naive@belief {cv(B):.4f}  spread@belief {cv(B,S):.4f}  "
          f"ORACLE@true {cv(T):.4f}   (≈equal -> position/certainty handling irrelevant here)")
    print("  CONCLUSION: with a well-localized robot (error << kernel), spreading/gating do "
          "not help; use naive. Spreading would only matter if localization were poor.")


if __name__ == "__main__":
    main()
