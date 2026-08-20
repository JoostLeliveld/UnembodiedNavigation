#!/usr/bin/env python3
"""What each candidate field says about the two competing routes.

Companion to offline_efe_solve.py. That script measures whether the runtime
objective ACTS on availability; this one measures whether the field CONTAINS the
signal at all. Kept as a separate artifact so the figure is rendered from disk
rather than recomputed inline.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import common as C  # noqa: E402

FIELDS = {
    "C2_operational_gp": C.REPO / "logs/visibility_comparison/spawn_grid_20260727/fused_planner_four_camera.npz",
    "C3_mono_depth": C.OUT_ROOT / "mono_depth_planner_v1/fused_planner_four_camera.npz",
    "C4_depth_plus_gp": C.OUT_ROOT / "depth_gp_planner_v1/fused_planner_four_camera.npz",
}
LABELS = {"C2_operational_gp": "Operational GP\n(needs survey)",
          "C3_mono_depth": "Monocular depth",
          "C4_depth_plus_gp": "Monocular depth + GP"}
CAMPAIGN = Path("/tmp/claude-1000/-home-joostleliveld-Thesis/fe7b3fd8-c80b-4a45-a8a5-49d94d23993c/scratchpad/e4_campaign.yaml")


def dense(p, step=0.05):
    p = np.asarray(p, float)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1); tot = seg.sum()
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, tot, max(2, int(tot / step)))
    return np.column_stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])])


def main() -> None:
    ap = C.build_apparatus()
    cfg = yaml.safe_load(CAMPAIGN.read_text())
    rows = []
    for task, tc in cfg["tasks"].items():
        routes = {r["name"]: dense(r["waypoints"]) for r in json.loads(tc["optimizer_initial_routes_json"])}
        for key, path in FIELDS.items():
            f = np.load(path)["P_conservative_plan_map"]
            blind = float(C.sample_field_at(f, ap.xs, ap.ys, routes["availability_blind"]).min())
            detour = float(C.sample_field_at(f, ap.xs, ap.ys, routes["cad_reference"]).min())
            rows.append(dict(task=task, field=key, label=LABELS[key].replace("\n", " "),
                             needs_survey=str(key == "C2_operational_gp").lower(),
                             blind_min=f"{blind:.6f}", detour_min=f"{detour:.6f}",
                             ratio=f"{detour/max(blind,1e-6):.3f}"))
            print(f"  {task:<24}{LABELS[key].replace(chr(10),' '):<26}{blind:>9.4f}{detour:>10.4f}{detour/max(blind,1e-6):>9.1f}x")
    out = C.OUT_ROOT / "e5_offline_efe_solve"
    C.write_csv(out / "route_field_discrimination.csv",
                ("task","field","label","needs_survey","blind_min","detour_min","ratio"), rows)
    print(f"\nwrote {out}/route_field_discrimination.csv")


if __name__ == "__main__":
    main()
