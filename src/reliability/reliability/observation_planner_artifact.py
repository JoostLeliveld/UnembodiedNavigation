"""P6: build planner-compatible p_use field artifacts, one per observability SOURCE.

The planner consumes observability as a gridded ``.npz`` visibility field
(`planning.core.visibility_gp_map.GPVisibilityMapModel`) → `make_prob_state_casadi` →
`expected_visibility_ca` → precision blend → R_plan. That whole adapter is held FIXED. The
only thing that differs between planner conditions is *which p_use field* is loaded:

    uniform    P0  constant = training usable rate           (no spatial info)
    geometry   P1  selected B2 FOV/range model               (calibration only, transferable)
    gp         P2  learned observability GP (diagnostic)      (does not beat geometry — P4)
    oracle     P4  in-sample empirical usable-rate grid       (best achievable field; EVAL-only)

Each artifact uses the identical grid and schema, so the planner adapter is byte-identical
across conditions — the required Gate-5 property. p_use is learned from operational belief
(x,y); no GT enters `uniform`/`geometry`/`gp`. `oracle` is the empirical field and is labelled
evaluation-only.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import pathlib
from typing import Any

import numpy as np

SOURCES = ("uniform", "geometry", "gp", "oracle")


@dataclass(frozen=True)
class FieldGrid:
    xs: np.ndarray
    ys: np.ndarray

    @property
    def XY(self) -> np.ndarray:
        XX, YY = np.meshgrid(self.xs, self.ys)  # (ny, nx)
        return np.column_stack([XX.ravel(), YY.ravel()])

    def reshape(self, flat: np.ndarray) -> np.ndarray:
        return np.asarray(flat, dtype=float).reshape(self.ys.shape[0], self.xs.shape[0])


def grid_from_dataframe(df, *, margin_m: float = 0.5, resolution_m: float = 0.25) -> FieldGrid:
    x0, x1 = float(df["state_x"].min()) - margin_m, float(df["state_x"].max()) + margin_m
    y0, y1 = float(df["state_y"].min()) - margin_m, float(df["state_y"].max()) + margin_m
    xs = np.arange(x0, x1 + resolution_m, resolution_m)
    ys = np.arange(y0, y1 + resolution_m, resolution_m)
    return FieldGrid(xs=xs, ys=ys)


def build_p_use_field(source: str, df, grid: FieldGrid, *, target: str = "usable_label",
                      gp_length_scale: float = 1.5) -> np.ndarray:
    """Return the p_use field on the grid (ny, nx) for one source."""
    from reliability.observation_baselines import FovRangeLogistic, GridFrequency
    from reliability.observation_gp import ObservabilityGP

    xy = df[["state_x", "state_y"]].to_numpy()
    y = df[target].to_numpy().astype(float)
    XY = grid.XY
    if source == "uniform":
        flat = np.full(len(XY), float(y.mean()))
    elif source == "geometry":
        flat = FovRangeLogistic().fit(xy, y).predict_proba(XY)
    elif source == "gp":
        flat = ObservabilityGP(length_scale=gp_length_scale).fit(xy, y).predict_proba(XY)
    elif source == "oracle":
        flat = GridFrequency(min_count=3).fit(xy, y).predict_proba(XY)
    else:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCES}")
    return np.clip(grid.reshape(flat), 1e-4, 1.0 - 1e-4)


def write_planner_artifact(
    path: str, grid: FieldGrid, p_field: np.ndarray, *,
    camera_pos: tuple[float, float, float], source: str, provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write the `.npz` in the exact GPVisibilityMapModel schema + a sidecar manifest."""
    p_field = np.clip(np.asarray(p_field, dtype=float), 1e-4, 1.0 - 1e-4)
    out = pathlib.Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        xs=grid.xs,
        ys=grid.ys,
        P_mean_map=p_field,
        P_conservative_plan_map=p_field,  # planner interpolates this; = mean (no extra shrink)
        F_mean_map=np.zeros_like(p_field),
        F_std_map=np.full_like(p_field, 0.05),
        camera_pos=np.asarray(camera_pos, dtype=float),
        target_height=np.asarray([0.0], dtype=float),
    )
    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    manifest = {
        "source": source,
        "artifact": str(out),
        "artifact_sha256": digest,
        "schema": "GPVisibilityMapModel/P_conservative_plan_map",
        "grid": {"nx": int(grid.xs.size), "ny": int(grid.ys.size),
                 "x_range": [float(grid.xs[0]), float(grid.xs[-1])],
                 "y_range": [float(grid.ys[0]), float(grid.ys[-1])]},
        "field_mean": float(p_field.mean()),
        "camera_pos": list(camera_pos),
        "ground_truth_used": source == "oracle",
        "provenance": provenance,
    }
    with open(str(out) + ".manifest.json", "w", encoding="utf-8") as h:
        json.dump(manifest, h, indent=2, default=str)
    return manifest
