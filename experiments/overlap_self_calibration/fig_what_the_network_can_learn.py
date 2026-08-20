"""One figure: which parts of a camera's mounting the network can measure itself."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import observability as ob  # noqa: E402

LABEL = {"pan_deg": "aim\nleft/right", "tilt_deg": "aim\nup/down", "roll_deg": "roll",
         "tx_m": "position\nE\u2013W", "ty_m": "position\nN\u2013S", "tz_m": "mount\nheight"}
LONG = {"pan_deg": "aim left/right (pan)", "tilt_deg": "aim up/down (tilt)", "roll_deg": "roll",
        "tx_m": "position east-west", "ty_m": "position north-south", "tz_m": "mount height"}
NUDGE = {"pan_deg": (7, -9), "tilt_deg": (7, -3), "roll_deg": (7, 4),
         "tx_m": (7, 5), "ty_m": (7, -8), "tz_m": (7, -3)}
ANG = ("pan_deg", "tilt_deg", "roll_deg")
POS = ("tx_m", "ty_m", "tz_m")
C_DATA, C_PRIOR, C_MID = "#1b6ca8", "#b0b7bd", "#e08a1e"


def main() -> None:
    obs = json.loads((ob.OUT / "observability.json").read_text())
    sen = json.loads((ob.OUT / "sensitivity.json").read_text())

    fig = plt.figure(figsize=(14.5, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.0, 1.0], wspace=0.30,
                          left=0.052, right=0.988, top=0.795, bottom=0.155)

    # --- panel 1: does the answer come from the data or from the prior? -----
    ax = fig.add_subplot(gs[0, 0])
    sweep = sen["prior_sensitivity"]
    for group, style in ((ANG, "-"), (POS, "--")):
        for k in group:
            xs = [v["prior_sd"][k] for v in sweep.values()]
            ys = [v["posterior_sd_mean_over_cameras"][k] for v in sweep.values()]
            xs = np.array(xs) / (1.0 if k in ANG else 0.05)
            ys = np.array(ys) / (1.0 if k in ANG else 0.05)
            flat = ys[-1] / ys[0] < 1.5
            ax.plot(xs, ys, style, lw=2.4 if flat else 1.6,
                    color=C_DATA if flat else C_MID if ys[-1] / xs[-1] < 0.2 else C_PRIOR,
                    marker="o", ms=4.5)
            ax.annotate(LONG[k], (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=NUDGE[k], fontsize=8.5, va="center",
                        color=C_DATA if flat else C_MID if ys[-1] / xs[-1] < 0.2 else "#7a8288")
    lim = np.array([0.2, 40.0])
    ax.plot(lim, lim, ":", color="#c3c9ce", lw=1.4, zorder=0)
    ax.annotate("learned nothing", (5.0, 6.6), fontsize=8.5, color="#9aa2a8",
                rotation=30, ha="center", rotation_mode="anchor")
    ax.set(xscale="log", yscale="log", xlim=(0.18, 130), ylim=(3e-3, 40))
    ax.set_xlabel("how uncertain we were before  (prior, in units of 1° / 5 cm)")
    ax.set_ylabel("how uncertain we are after  (same units)")
    ax.set_title("A line that flattens was measured.\nA line on the diagonal was assumed.",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, which="both", lw=0.5)

    # --- panel 2: overlap is what does the work ------------------------------
    ax = fig.add_subplot(gs[0, 1])
    keys = ANG + POS
    arms = [("sightings only one camera saw\n(2255 sites)", "single_camera_sites_only", C_PRIOR),
            ("sightings two cameras saw at once\n(921 sites)", "overlap_only", C_DATA)]
    x = np.arange(len(keys))
    for i, (name, arm, col) in enumerate(arms):
        per = obs["arms"][arm]["per_camera"]
        vals = [np.mean([per[c][k]["posterior_sd"] for c in per]) /
                (1.0 if k in ANG else 0.05) for k in keys]
        ax.bar(x + (i - 0.5) * 0.38, vals, 0.36, label=name, color=col,
               edgecolor="white", lw=0.6)
    ax.axhline(1.0, color="#444", lw=1.1, ls=":")
    ax.annotate("what we assumed before looking", (len(keys) - 0.4, 1.06), fontsize=8.2,
                color="#444", ha="right")
    ax.set(xticks=x, yscale="log", ylim=(3e-3, 3))
    ax.set_xticklabels([LABEL[k] for k in keys], fontsize=8.0)
    ax.set_ylabel("uncertainty left  (units of 1° / 5 cm)")
    ax.legend(fontsize=8.3, loc="lower left", framealpha=0.95)
    ax.set_title("Only where two cameras see the same spot\ndoes the network learn its own aim.",
                 fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, axis="y", which="both", lw=0.5)

    # --- panel 3: crossing angle, at equal site count ------------------------
    ax = fig.add_subplot(gs[0, 2])
    cm = sen["count_matched_crossing_angle"]
    vals = cm["posterior_sd_mean_over_cameras"]
    for i, (lab, key, col) in enumerate(
            (("cameras %d° apart" % round(cm["narrow_pairs_median_separation_deg"]),
              "narrow", C_PRIOR),
             ("cameras %d° apart" % round(cm["wide_pairs_median_separation_deg"]),
              "wide_median_of_draws", C_DATA))):
        v = [vals[k][key] / (1.0 if k in ANG else 0.05) for k in keys]
        ax.bar(x + (i - 0.5) * 0.38, v, 0.36, label=lab, color=col,
               edgecolor="white", lw=0.6)
    ax.set(xticks=x, yscale="log", ylim=(3e-3, 3))
    ax.set_xticklabels([LABEL[k] for k in keys], fontsize=8.0)
    ax.set_ylabel("uncertainty left  (units of 1° / 5 cm)")
    ax.legend(fontsize=8.3, loc="lower left", framealpha=0.95)
    ax.set_title("Facing each other beats side by side\n"
                 "— %d sites each, so this is angle, not count."
                 % cm["sites_per_arm"], fontsize=10.5, loc="left")
    ax.grid(alpha=0.25, axis="y", which="both", lw=0.5)

    fig.suptitle("Four cameras watching the same robot can measure their own aim to a hundredth "
                 "of a degree —\nbut never their own position, because sliding the whole network "
                 "sideways looks like nothing at all.",
                 fontsize=12.6, x=0.055, ha="left", y=0.965)

    for ext in ("png", "pdf"):
        dest = ob.OUT / f"fig_what_the_network_can_learn.{ext}"
        fig.savefig(dest, dpi=180)
    print("wrote", ob.OUT / "fig_what_the_network_can_learn.png")


if __name__ == "__main__":
    main()
