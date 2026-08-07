#!/usr/bin/env python3
"""F04 renderer: calibration drift lifecycle -- detection before harm (v3 scope).

This script PLOTS ONLY. It reads the audited `drift_lifecycle.json` produced by
`experiments/calibration_drift_lifecycle/drift_lifecycle.py` and renders it. It
refits nothing, resamples nothing, smooths nothing and imports no runtime code.
Every plotted value is a literal number from the JSON.

The result the figure carries (research/08_figures.md, row F04, allowed claim):

    In the explicit v3 controlled-injection scope, the change statistic detects
    camera-C yaw drift at 0.1 degrees before first harm at 0.25 degrees.

Two statistics live in the JSON, both GT-free and both built from the same
operational cross-axis residual with the camera under test held out of its own
reference:

    absolute  |b_cross(d)|                / sigma_cross(d)  >= 1.2   (the commissioning gate)
    change    |b_cross(d) - b_cross(0)|   / sigma_cross(0)  >= 1.2   (the in-service monitor)

The absolute form is the deployed commissioning gate and CANNOT be reused in
service: it fires at rest on the cameras it left RAW, and it is non-monotone
(camera B's ratio *falls* from 5.02 to 0.31 at 0.25 deg as the injected drift
cancels its resident bias). Per the F04 spec it therefore appears only as a
separately labelled grey diagnostic (fig_d2), never as the headline evidence.

Ground truth (the `oracle_*` fields) is EVALUATION ONLY -- it scores whether the
stale correction helps or harms. It is never an input to either detector.

Outputs -> logs/studies/calibration_drift_lifecycle/exp1_stale_correction/
    fig_d1_drift_lifecycle.{pdf,png}          the claim: detection before harm
    fig_d2_absolute_gate_diagnostic.{pdf,png} the diagnostic: why change != absolute
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(_HERE)
OUT = REPO / "logs" / "studies" / "calibration_drift_lifecycle" / "exp1_stale_correction"
SOURCE = OUT / "drift_lifecycle.json"

# The audited input, research/08_figures.md row F04. A different hash means the
# study was regenerated and this figure's claim must be re-checked, not re-rendered.
SOURCE_SHA256 = "eecc912a1bb387da1afc2c00e1e00d96ff776827746d95be23e4fb69c28c2b4a"

# Fixed camera order and colours (F04 spec item 2). Okabe-Ito, colourblind-safe;
# marker and linestyle are redundant with colour so hue is never the only channel.
CAMERAS = ("camera_A", "camera_B", "camera_C", "camera_D")
CAM_COLOR = {
    "camera_A": "#0072B2",  # blue
    "camera_B": "#E69F00",  # orange
    "camera_C": "#009E73",  # bluish green
    "camera_D": "#CC79A7",  # reddish purple
}
CAM_MARKER = {"camera_A": "o", "camera_B": "s", "camera_C": "^", "camera_D": "D"}
# A and C track each other almost exactly on the change statistic, so they must not
# share a linestyle.
CAM_LS = {"camera_A": "-", "camera_B": "--", "camera_C": ":", "camera_D": "-."}

C_RAW = "#555555"        # uncorrected arm
C_DETECT = "#0072B2"     # detection annotation
C_HARM = "#D55E00"       # harm annotation
C_DIAG = "#7F7F7F"       # grey diagnostic

# Annotations sit over gridlines and rung markers; keep them readable in print.
BBOX = dict(facecolor="white", alpha=0.82, edgecolor="none", pad=1.5)

# The claim of record, asserted against the JSON before anything is drawn.
CLAIM_CAMERA = "camera_C"
CLAIM_DETECT_DEG = 0.1
CLAIM_HARM_DEG = 0.25


def _footer(fig, paragraphs: list[str]) -> None:
    """Scope/caveat block. Wrapped to a fixed column: with bbox_inches='tight' an
    over-long line silently widens the saved canvas and squashes the panels."""
    text = "\n".join(textwrap.fill(p, width=150) for p in paragraphs)
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=7.8,
             color="#333333", linespacing=1.5)


def _style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "axes.grid": True, "grid.color": "#CCCCCC",
        "grid.alpha": 0.45, "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "font.size": 9, "legend.fontsize": 8,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    })


def load() -> dict:
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise AssertionError(
            f"{SOURCE} hash {digest} != audited {SOURCE_SHA256}; the study was "
            "regenerated, so re-audit the claim before re-rendering F04."
        )
    return payload


def ladder(payload: dict, camera: str, field: str) -> dict:
    """The single summary entry for one camera and one fault axis."""
    hits = [s for s in payload["summary"]
            if s["camera"] == camera and s["drift_field"] == field]
    if len(hits) != 1:
        raise AssertionError(f"expected 1 summary entry for {camera}/{field}, got {len(hits)}")
    return hits[0]


def assert_inputs(payload: dict) -> dict:
    """F04 spec item 4. Any failure stops the build."""
    checks: dict[str, object] = {}

    identity = payload["identity_check_max_error_m"]
    if identity != 0:
        raise AssertionError(f"identity_check_max_error_m == {identity}, expected 0")
    checks["identity_check_max_error_m"] = identity
    checks["identity_check_detections"] = payload["identity_check_detections"]

    calib = payload["calibration"]
    if "projection_calibration_v3" not in calib:
        raise AssertionError(f"calibration {calib!r} is not the audited v3 input")
    checks["calibration"] = calib
    checks["source_sha256"] = SOURCE_SHA256

    threshold = payload["gate_threshold"]
    if threshold != 1.2:
        raise AssertionError(f"gate_threshold == {threshold}, expected 1.2")
    checks["gate_threshold"] = threshold

    claim = ladder(payload, CLAIM_CAMERA, "yaw_deg")
    if claim["drift_detected_at_change"] != CLAIM_DETECT_DEG:
        raise AssertionError(
            f"{CLAIM_CAMERA} change detection == {claim['drift_detected_at_change']}, "
            f"expected {CLAIM_DETECT_DEG}"
        )
    if claim["stale_correction_harmful_at"] != CLAIM_HARM_DEG:
        raise AssertionError(
            f"{CLAIM_CAMERA} first harm == {claim['stale_correction_harmful_at']}, "
            f"expected {CLAIM_HARM_DEG}"
        )
    checks["camera_C_detected_at_change_deg"] = claim["drift_detected_at_change"]
    checks["camera_C_harmful_at_deg"] = claim["stale_correction_harmful_at"]

    # The harm rung must also be readable straight off the plotted RMS curves,
    # not merely asserted by a summary field.
    crossings = [r["drift"] for r in claim["ladder"] if r["rms_stale_m"] > r["rms_raw_m"]]
    if not crossings or min(crossings) != CLAIM_HARM_DEG:
        raise AssertionError(
            f"plotted {CLAIM_CAMERA} RMS curves cross at {crossings}, expected first "
            f"crossing at {CLAIM_HARM_DEG}"
        )
    if CLAIM_DETECT_DEG >= CLAIM_HARM_DEG:
        raise AssertionError("detection rung is not strictly before the harm rung")
    checks["ordering"] = f"detect {CLAIM_DETECT_DEG} deg < harm {CLAIM_HARM_DEG} deg"

    # Panel C and panel D titles state negative results; enforce them too.
    d_yaw = ladder(payload, "camera_D", "yaw_deg")
    if any(r["rms_stale_m"] > r["rms_raw_m"] for r in d_yaw["ladder"]):
        raise AssertionError("camera_D yaw curves cross; panel C caption is wrong")
    checks["camera_D_yaw_harm"] = None
    tx_harm = {c: ladder(payload, c, "tx_m")["stale_correction_harmful_at"] for c in CAMERAS}
    if any(v is not None for v in tx_harm.values()):
        raise AssertionError(f"translation harm present {tx_harm}; panel D caption is wrong")
    if any(r["rms_stale_m"] > r["rms_raw_m"]
           for c in CAMERAS for r in ladder(payload, c, "tx_m")["ladder"]):
        raise AssertionError("a translation ladder crosses over; panel D caption is wrong")
    checks["translation_harm_any_camera"] = None

    # Per-camera detection counts, reported rather than assumed.
    checks["detections_per_camera"] = {
        cam: sorted({r["oracle_n"] for r in payload["rows"] if r["camera"] == cam})[0]
        for cam in CAMERAS
    }
    return checks


def _rung_axis(ax, rungs: list[float], label: str, fmt: str) -> None:
    """Ladder rungs are plotted at equal spacing: the fault ladder is ordinal, and
    a linear axis would crush the low rungs where the whole result lives."""
    ax.set_xticks(range(len(rungs)))
    ax.set_xticklabels([fmt.format(v) for v in rungs])
    ax.set_xlim(-0.25, len(rungs) - 0.75)
    ax.set_xlabel(label)


def _symlog_z(ax) -> None:
    """change_z spans 0 to 38; linear hides the 1.2 crossing, log cannot hold the
    exact 0 at rest. symlog with the break AT the threshold shows both."""
    ax.set_yscale("symlog", linthresh=1.2, linscale=1.1)
    ax.set_yticks([0, 0.5, 1.2, 2, 5, 10, 20, 40])
    ax.set_yticklabels(["0", "0.5", "1.2", "2", "5", "10", "20", "40"])
    ax.set_ylim(-0.12, 60)


def _mark_rungs(ax, rungs: list[float], detect: float | None, harm: float | None,
                detect_label: str, harm_label: str) -> None:
    if detect is not None:
        ax.axvline(rungs.index(detect), color=C_DETECT, ls=":", lw=1.6, zorder=1.5,
                   label=detect_label)
    if harm is not None:
        ax.axvline(rungs.index(harm), color=C_HARM, ls="--", lw=1.6, zorder=1.5,
                   label=harm_label)


def panel_rms(ax, entry: dict, rungs: list[float]) -> None:
    """Oracle RMS, raw vs stale-v3, for one CALIBRATE camera. Ground truth is used
    here for EVALUATION ONLY -- it scores the policy, it feeds no detector."""
    x = list(range(len(rungs)))
    raw = [r["rms_raw_m"] for r in entry["ladder"]]
    stale = [r["rms_stale_m"] for r in entry["ladder"]]
    cam = entry["camera"]

    harmful = [s > r for s, r in zip(stale, raw)]
    if any(harmful):
        ax.fill_between(x, raw, stale, where=harmful, interpolate=True, color=C_HARM,
                        alpha=0.25, lw=0, zorder=1.2,
                        label="stale correction worse than raw")
    ax.plot(x, raw, color=C_RAW, ls="--", lw=1.7, marker="o", ms=5.5, mfc="white",
            mew=1.4, zorder=3, label="raw (no correction)")
    ax.plot(x, stale, color=CAM_COLOR[cam], ls="-", lw=2.0, marker=CAM_MARKER[cam],
            ms=6.0, zorder=3, label="stale v3 correction (fitted at commissioning)")
    ax.set_ylabel("oracle RMS (m)  [GT, evaluation only]")


def fig_d1(payload: dict, checks: dict) -> Path:
    yaw = [float(v) for v in payload["yaw_drift_deg"]]
    tx = [float(v) for v in payload["translation_drift_m"]]

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.6))
    (ax_a, ax_b), (ax_c, ax_d) = axes

    cam_c = ladder(payload, "camera_C", "yaw_deg")
    cam_d = ladder(payload, "camera_D", "yaw_deg")

    # -- Panel A: the claim. Camera C, raw vs stale, harm crossover. -------------
    panel_rms(ax_a, cam_c, yaw)
    _mark_rungs(ax_a, yaw, CLAIM_DETECT_DEG, CLAIM_HARM_DEG,
                f"change detector fires ({CLAIM_DETECT_DEG:g} deg)",
                f"first harm ({CLAIM_HARM_DEG:g} deg)")
    _rung_axis(ax_a, yaw, "injected yaw drift (deg), ladder rungs equally spaced", "{:g}")
    ax_a.set_title("A  camera C (commissioned CALIBRATE) - the stale v3 correction inverts\n"
                   "controlled yaw injection on one held-out capture, calibration v3",
                   loc="left")
    rest, harm_rung = cam_c["ladder"][0], cam_c["ladder"][yaw.index(CLAIM_HARM_DEG)]
    ax_a.annotate(f"commissioning win: halves error at rest\n"
                  f"{rest['rms_raw_m']:.3f} -> {rest['rms_stale_m']:.3f} m",
                  xy=(0, rest["rms_stale_m"]), xytext=(0.30, 0.185), fontsize=8,
                  color=CAM_COLOR["camera_C"], bbox=BBOX,
                  arrowprops=dict(arrowstyle="->", color=CAM_COLOR["camera_C"], lw=1.1))
    ax_a.annotate(f"inverted: {harm_rung['rms_stale_m']:.3f} m corrected\n"
                  f"vs {harm_rung['rms_raw_m']:.3f} m raw",
                  xy=(2, harm_rung["rms_stale_m"]), xytext=(2.55, 0.055), fontsize=8,
                  color=C_HARM, bbox=BBOX,
                  arrowprops=dict(arrowstyle="->", color=C_HARM, lw=1.1))
    ax_a.legend(loc="upper left", framealpha=0.95)

    # -- Panel B: the detector. change_z, all cameras, yaw. ---------------------
    x = list(range(len(yaw)))
    for cam in CAMERAS:
        entry = ladder(payload, cam, "yaw_deg")
        z = [r["change_z"] for r in entry["ladder"]]
        ax_b.plot(x, z, color=CAM_COLOR[cam], ls=CAM_LS[cam], lw=1.8,
                  marker=CAM_MARKER[cam], ms=5.5, zorder=3,
                  label=f"{cam[-1]} ({entry['commissioned_policy']})")
        hit = entry["drift_detected_at_change"]
        if hit is not None:
            i = yaw.index(hit)
            ax_b.plot([i], [z[i]], marker="o", ms=13, mfc="none",
                      mec=CAM_COLOR[cam], mew=1.8, zorder=4)
    ax_b.axhline(1.2, color="black", lw=1.4, ls="-", zorder=2)
    ax_b.text(len(yaw) - 1.05, 1.32, "threshold 1.2", fontsize=8, ha="right", va="bottom")
    _mark_rungs(ax_b, yaw, CLAIM_DETECT_DEG, CLAIM_HARM_DEG, None, None)
    ax_b.text(yaw.index(CLAIM_DETECT_DEG) - 0.08, 52, "detected", color=C_DETECT,
              fontsize=8.2, ha="right", va="center", fontweight="bold", bbox=BBOX)
    ax_b.text(yaw.index(CLAIM_HARM_DEG) + 0.08, 52, "harm begins (camera C)", color=C_HARM,
              fontsize=8.2, ha="left", va="center", fontweight="bold", bbox=BBOX)
    ax_b.annotate("", xy=(yaw.index(CLAIM_HARM_DEG), 24), xytext=(yaw.index(CLAIM_DETECT_DEG), 24),
                  arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
    ax_b.text(1.5, 28.5, "one rung of margin", fontsize=8.2, ha="center", color="#333333",
              bbox=BBOX)
    _symlog_z(ax_b)
    _rung_axis(ax_b, yaw, "injected yaw drift (deg), ladder rungs equally spaced", "{:g}")
    ax_b.set_ylabel("change statistic  $|b(\\delta)-b(0)|\\,/\\,\\sigma(0)$   (GT-free)")
    ax_b.set_title("B  the change statistic is monotone and fires before harm\n"
                   "rings = first rung detected; A, B, C at 0.1 deg, D at 0.25 deg",
                   loc="left")
    ax_b.legend(loc="lower right", ncol=2, title="camera (commissioned policy)",
                title_fontsize=8, framealpha=0.95)

    # -- Panel C: the counterexample. Camera D never crosses. -------------------
    panel_rms(ax_c, cam_d, yaw)
    d_detect = cam_d["drift_detected_at_change"]
    _mark_rungs(ax_c, yaw, d_detect, None,
                f"change detector fires ({d_detect:g} deg for D)", None)
    _rung_axis(ax_c, yaw, "injected yaw drift (deg), ladder rungs equally spaced", "{:g}")
    ax_c.set_title("C  camera D (also CALIBRATE) never crosses over - harm is not universal\n"
                   "its commissioned correction is small, so drift swamps both arms",
                   loc="left")
    ax_c.annotate("corrected arm stays below raw at every rung:\n"
                  "no expiry to detect on this camera",
                  xy=(4, cam_d["ladder"][4]["rms_stale_m"]), xytext=(1.25, 0.44),
                  fontsize=8, color="#333333", bbox=BBOX,
                  arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1))
    ax_c.legend(loc="upper left", framealpha=0.95)

    # -- Panel D: the second fault axis. change_z, translation. ----------------
    xt = list(range(len(tx)))
    for cam in CAMERAS:
        entry = ladder(payload, cam, "tx_m")
        z = [r["change_z"] for r in entry["ladder"]]
        ax_d.plot(xt, z, color=CAM_COLOR[cam], ls=CAM_LS[cam], lw=1.8,
                  marker=CAM_MARKER[cam], ms=5.5, zorder=3, label=cam[-1])
        hit = entry["drift_detected_at_change"]
        if hit is not None:
            i = tx.index(hit)
            ax_d.plot([i], [z[i]], marker="o", ms=13, mfc="none",
                      mec=CAM_COLOR[cam], mew=1.8, zorder=4)
    ax_d.axhline(1.2, color="black", lw=1.4, ls="-", zorder=2)
    ax_d.text(len(tx) - 1.05, 1.32, "threshold 1.2", fontsize=8, ha="right", va="bottom")
    ax_d.axvline(tx.index(0.025), color=C_DETECT, ls=":", lw=1.6, zorder=1.5,
                 label="first detection (0.025 m)")
    _symlog_z(ax_d)
    _rung_axis(ax_d, tx, "injected translation drift $t_x$ (m), rungs equally spaced", "{:g}")
    ax_d.set_ylabel("change statistic (GT-free)")
    ax_d.set_title("D  second fault axis: translation. Detected at 0.025-0.05 m;\n"
                   "NO camera is harmed by its stale correction at any translation rung",
                   loc="left")
    ax_d.legend(loc="lower right", ncol=2, title="camera", title_fontsize=8,
                framealpha=0.95)

    n = checks["detections_per_camera"]
    fig.suptitle(
        "Calibration drift lifecycle (F04): a GT-free change statistic detects yaw drift "
        "one rung before the stale correction becomes harmful",
        fontsize=12.5, fontweight="bold", y=0.995)
    _footer(fig, [
        "Scope: CONTROLLED step-fault injection on ONE held-out capture "
        "(fusion_handover_20260721, held out of the v3 calibration fit), calibration v3. "
        f"Identity drift reproduces the deployed projection to 0.00e+00 m over "
        f"{checks['identity_check_detections']} detections. Gate/oracle detections per "
        f"camera: A {n['camera_A']}, B {n['camera_B']}, C {n['camera_C']}, D {n['camera_D']}.",
        "Independent unit: unique camera site; deterministic fault ladder, one magnitude "
        "per whole capture (a step, not a ramp). Rungs are the ONLY measured points; "
        "segments join them and imply no interpolation. No confidence band is drawn.",
        "This measures a detection THRESHOLD IN DRIFT MAGNITUDE. It is NOT a real-drift "
        "detection latency, NOT an in-service false-alarm rate, and NOT current policy. "
        "Ground truth (oracle RMS) scores the outcome only; neither detector sees it.",
    ])

    fig.tight_layout(rect=(0, 0.105, 1, 0.965))
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"fig_d1_drift_lifecycle.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths[0]


def fig_d2(payload: dict, checks: dict) -> Path:
    """The absolute commissioning gate, as a grey diagnostic only (F04 spec item 2)."""
    yaw = [float(v) for v in payload["yaw_drift_deg"]]
    tx = [float(v) for v in payload["translation_drift_m"]]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.4, 5.4))

    for ax, rungs, field, xlabel, fmt in (
        (ax_l, yaw, "yaw_deg", "injected yaw drift (deg), rungs equally spaced", "{:g}"),
        (ax_r, tx, "tx_m", "injected translation drift $t_x$ (m), rungs equally spaced", "{:g}"),
    ):
        x = list(range(len(rungs)))
        for cam in CAMERAS:
            entry = ladder(payload, cam, field)
            ratio = [r["gate_ratio"] for r in entry["ladder"]]
            ax.plot(x, ratio, color=CAM_COLOR[cam], ls=CAM_LS[cam], lw=1.8,
                    marker=CAM_MARKER[cam], ms=5.5, zorder=3,
                    label=f"{cam[-1]} ({entry['commissioned_policy']})")
        ax.axhline(1.2, color="black", lw=1.4, zorder=2)
        ax.text(len(rungs) - 1.05, 1.35, "threshold 1.2", fontsize=8, ha="right", va="bottom")
        ax.axhspan(0, 1.2, color=C_DIAG, alpha=0.13, lw=0, zorder=0)
        ax.set_yscale("symlog", linthresh=1.2, linscale=1.1)
        ax.set_yticks([0, 0.5, 1.2, 2, 5, 10, 20, 40])
        ax.set_yticklabels(["0", "0.5", "1.2", "2", "5", "10", "20", "40"])
        ax.set_ylim(-0.05, 60)
        _rung_axis(ax, rungs, xlabel, fmt)
        ax.set_ylabel("absolute gate ratio  $|b(\\delta)|\\,/\\,\\sigma(\\delta)$")
        ax.text(0.02, 0.055, "gate PASSES (reads healthy)", transform=ax.transAxes,
                fontsize=7.6, color="#555555", style="italic")

    b_yaw = ladder(payload, "camera_B", "yaw_deg")["ladder"]
    b_tx = ladder(payload, "camera_B", "tx_m")["ladder"]
    a_rest = ladder(payload, "camera_A", "yaw_deg")["ladder"][0]["gate_ratio"]

    ax_l.annotate(f"A and B already FIRE at zero drift\n"
                  f"({a_rest:.1f} and {b_yaw[0]['gate_ratio']:.1f}): a standing false alarm",
                  xy=(0, a_rest), xytext=(0.5, 25), fontsize=8.2, color="#333333",
                  bbox=BBOX, arrowprops=dict(arrowstyle="->", color="#333333", lw=1.1))
    ax_l.annotate(f"MASKED: B falls to {b_yaw[2]['gate_ratio']:.2f} at 0.25 deg\n"
                  f"and PASSES the gate while drifted",
                  xy=(2, b_yaw[2]["gate_ratio"]), xytext=(2.35, 0.10), fontsize=8.2,
                  color=C_HARM, bbox=BBOX,
                  arrowprops=dict(arrowstyle="->", color=C_HARM, lw=1.2))
    ax_l.set_title("yaw: non-monotone, and already tripped at rest", loc="left")
    ax_r.annotate(f"MASKED again: B collapses to\n{b_tx[3]['gate_ratio']:.3f} at 5 cm",
                  xy=(3, b_tx[3]["gate_ratio"]), xytext=(1.35, 0.09), fontsize=8.2,
                  color=C_HARM, bbox=BBOX,
                  arrowprops=dict(arrowstyle="->", color=C_HARM, lw=1.2))
    ax_r.set_title("translation: the same two failure modes", loc="left")

    handles, labels = ax_l.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.915), ncol=4,
               title="camera (commissioned policy) - curve is that camera's deployed stream",
               title_fontsize=8, framealpha=0.95)

    fig.suptitle(
        "DIAGNOSTIC (not manuscript evidence): the absolute commissioning gate cannot "
        "double as the in-service drift monitor",
        fontsize=12.0, fontweight="bold", y=0.995)
    fig.text(
        0.5, 0.005,
        "Same audited JSON, same controlled v3 injection, same held-out capture as F04 "
        "fig_d1; curves are the commissioned-policy stream per camera. Two independent "
        "failure modes: (1) the ratio already exceeds 1.2 at zero drift on the cameras "
        "commissioned RAW, so an absolute in-service alarm would be permanently tripped; "
        "(2) it is non-monotone, because injected drift can cancel a camera's resident "
        "bias, so a drifted camera can read SAFER than at rest.\nThe specific rest-state "
        "values are capture-unstable (camera B rests on 15 detections) and must not be "
        "quoted as properties of these cameras -- only as evidence that the absolute "
        "statistic is capture-unstable. This panel motivates the change statistic; it "
        "states no detection performance.",
        ha="center", va="bottom", fontsize=7.8, color="#333333", linespacing=1.5)

    fig.tight_layout(rect=(0, 0.135, 1, 0.94))
    paths = []
    for ext in ("pdf", "png"):
        p = OUT / f"fig_d2_absolute_gate_diagnostic.{ext}"
        fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    return paths[0]


def main() -> int:
    _style()
    payload = load()
    checks = assert_inputs(payload)
    fig_d1(payload, checks)
    fig_d2(payload, checks)
    print(json.dumps({"asserted": checks}, indent=2))
    print("wrote ->", OUT / "fig_d1_drift_lifecycle.{pdf,png}")
    print("wrote ->", OUT / "fig_d2_absolute_gate_diagnostic.{pdf,png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
