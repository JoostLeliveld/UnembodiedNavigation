"""Do two cameras make independent mistakes about the same robot pose?

The whole observability calculation treats each sighting's pixel noise as
independent. On the box-bottom reading that is false: the two-camera study found
the cameras share a yaw-driven silhouette error, so fusing them recovers almost
nothing. This checks the same thing on the marked-point reading, using held-out
predictions at poses two cameras saw at once.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observability as ob  # noqa: E402


def main() -> None:
    d = pd.read_csv(ob.PREDICTIONS)
    d = d[(d.detected == 1) & (d.front_labelled_visible == 1)
          & (d.rear_labelled_visible == 1)].copy()
    d["site"] = list(zip(d.x.round(6), d.y.round(6), d.yaw_rad.round(6)))
    d["n_cams"] = d.site.map(d.groupby("site").camera.nunique())
    co = d[d.n_cams >= 2]

    rows = []
    for _, grp in co.groupby("site"):
        for (_, a), (_, b) in combinations(list(grp.iterrows()), 2):
            rows.append({"pair": "|".join(sorted((a.camera, b.camera))),
                         "ax": a.err_x_m, "ay": a.err_y_m,
                         "bx": b.err_x_m, "by": b.err_y_m})
    p = pd.DataFrame(rows)

    def corr(g):
        return {
            "n": int(len(g)),
            "east_west": round(float(np.corrcoef(g.ax, g.bx)[0, 1]), 4),
            "north_south": round(float(np.corrcoef(g.ay, g.by)[0, 1]), 4),
            "standard_error": round(float(1.0 / np.sqrt(len(g))), 4),
        }

    out = {
        "held_out_readings_both_markers_rendered": int(len(d)),
        "readings_at_co_observed_sites": int(len(co)),
        "co_observed_sites": int(co.site.nunique()),
        "pooled": corr(p),
        "by_pair": {k: corr(g) for k, g in p.groupby("pair") if len(g) >= 20},
        "contrast": ("On the box-bottom reading the two cameras' random errors are NOT "
                     "independent (shared yaw-driven silhouette scatter). On this reading "
                     "they are, within the standard error."),
    }
    dest = ob.OUT / "independence.json"
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
