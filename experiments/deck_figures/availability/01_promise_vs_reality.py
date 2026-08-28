"""Slide 3: what the camera layout promises, and what it delivers."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src
import style as D

OUT = D.REPO / "logs/studies/deck_figures/availability"; OUT.mkdir(parents=True, exist_ok=True)
lay = D.layout()
geo = src.geometric_field()
use, _ = src.support_field()
gv = np.array([geo[k] for k in sorted(geo)])
uv = np.array([use[k] for k in sorted(geo)])

fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.4), constrained_layout=True)
for ax, field, title, sub in (
    (axes[0], geo, "What the camera layout promises",
     f"every one of the {len(geo)} floor positions has a clear line of sight\n"
     f"to at least one camera, from every heading"),
    (axes[1], use, "What the cameras actually deliver",
     f"{np.mean(uv == 0)*100:.0f}% of the same positions never produced a usable\n"
     f"sighting from any camera"),
):
    D.draw_warehouse(ax, lay)
    sm = D.draw_support(ax, field, hatch_zero=(field is use))
    ax.set_title(title, loc="left", fontsize=18, color=D.INK, pad=40)
    ax.text(0.0, 1.005, sub, transform=ax.transAxes, fontsize=12.5,
            color=D.INK2, va="bottom", ha="left")

cb = fig.colorbar(sm, ax=axes, shrink=0.6, pad=0.012, aspect=26)
cb.set_label("chance of a usable camera sighting", color=D.INK2, fontsize=12)
cb.outline.set_edgecolor("#d5d4cf")
fig.text(0.012, 0.02,
         "Orange hatching marks positions where no camera ever gave a usable sighting.  "
         "2 316 robot placements: 386 floor positions x 6 headings, all five cameras at each.",
         fontsize=11.5, color=D.INK2)
fig.savefig(OUT / "01_promise_vs_reality.png", dpi=190, bbox_inches="tight")
print("wrote 01_promise_vs_reality.png")
