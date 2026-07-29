#!/usr/bin/env python3
"""P6: build the four planner p_use field artifacts (uniform/geometry/gp/oracle).

    python3 scripts/reliability/build_planner_p_use_artifacts.py \
        --dataset logs/studies/usable_observation/dataset_v1/observations.parquet \
        --output  logs/studies/usable_observation/planner_conditions_v1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _pkg in (_ROOT / "src").glob("*"):
    if (_pkg / _pkg.name).is_dir():
        sys.path.insert(0, str(_pkg))

from reliability.observation_baselines import AWS_CAM_POS  # noqa: E402
from reliability.observation_planner_artifact import (  # noqa: E402
    SOURCES,
    build_p_use_field,
    grid_from_dataframe,
    write_planner_artifact,
)


def _figure(artifacts, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(artifacts), figsize=(4.6 * len(artifacts), 3.6))
    for ax, (source, npz) in zip(np.atleast_1d(axes), artifacts.items()):
        d = np.load(npz)
        im = ax.pcolormesh(d["xs"], d["ys"], d["P_conservative_plan_map"], cmap="viridis",
                           vmin=0, vmax=1, shading="auto")
        ax.set_title(f"p_use source: {source}")
        ax.set_xlabel("belief x [m]"); ax.set_ylabel("belief y [m]")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    p = out / "p_use_fields.png"; fig.savefig(p, dpi=130)
    return str(p)


def main() -> int:
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="logs/studies/usable_observation/dataset_v1/observations.parquet")
    ap.add_argument("--output", default="logs/studies/usable_observation/planner_conditions_v1")
    ap.add_argument("--target", default="usable_label")
    args = ap.parse_args()

    df = pd.read_parquet(args.dataset)
    out = pathlib.Path(args.output); out.mkdir(parents=True, exist_ok=True)
    grid = grid_from_dataframe(df)

    manifests = {}
    artifacts = {}
    for source in SOURCES:
        field = build_p_use_field(source, df, grid, target=args.target)
        npz = out / f"p_use_{source}.npz"
        manifests[source] = write_planner_artifact(
            str(npz), grid, field, camera_pos=AWS_CAM_POS, source=source,
            provenance={"dataset": args.dataset, "target": args.target, "n_rows": int(len(df))},
        )
        artifacts[source] = str(npz)

    fig = _figure(artifacts, out)
    summary = {
        "grid": manifests["uniform"]["grid"],
        "adapter": "FIXED: GPVisibilityMapModel + expected_visibility_ca + precision blend; only the field differs",
        "sources": {s: {"artifact": m["artifact"], "sha256": m["artifact_sha256"],
                        "field_mean": m["field_mean"], "ground_truth_used": m["ground_truth_used"]}
                    for s, m in manifests.items()},
        "field_figure": fig,
    }
    with open(out / "planner_conditions_manifest.json", "w", encoding="utf-8") as h:
        json.dump(summary, h, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
