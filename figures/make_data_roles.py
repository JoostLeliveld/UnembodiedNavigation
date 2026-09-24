#!/usr/bin/env python3
"""The reference dataset v8: every captured robot position by the role it plays.

D_mu trains the correction, D_R fits the covariance models, D_dev compares and checks them,
and the final audit is sealed until the models are frozen. Read through pipeline/dataset.py,
the only loader.

    python3 figures/make_data_roles.py
"""
from __future__ import annotations

from collections import Counter

import numpy as np

import paper as P
from pipeline import dataset

ROLES = (("D_mu", "correction ($D_\\mu$)", "#56B4E9", "o"),
         ("D_R", "covariance ($D_R$)", "#CC79A7", "o"),
         ("D_dev", "development ($D_\\mathrm{dev}$)", "#E69F00", "o"),
         ("final_audit", "final audit (sealed)", P.INK, "s"))


def main():
    positions = {}
    for r in dataset.load_rows():
        positions.setdefault(r["position_key"], (float(r["robot_x"]), float(r["robot_y"]), r["stratum"]))
    counts = Counter(role for _, _, role in positions.values())
    fig, ax = P.plt.subplots(figsize=(P.COLUMN, 2.75))
    P.draw_map(ax)
    for role, label, colour, marker in ROLES:
        xy = np.array([(x, y) for x, y, r in positions.values() if r == role])
        ax.scatter(xy[:, 0], xy[:, 1], s=3.2 if role != "final_audit" else 5, marker=marker,
                   c=colour, lw=0, zorder=6, label=f"{label}: {counts[role]}")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=6.3, markerscale=2.0,
              handletextpad=0.2, columnspacing=1.0)
    ax.set_title(f"{len(positions)} positions, 20 opportunities each (5 cameras × 4 headings)", fontsize=7, pad=2)
    P.save(fig, "data_collection_roles")
    print(dict(counts))


if __name__ == "__main__":
    main()
