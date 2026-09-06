#!/usr/bin/env python3
"""Build the approved research plan and vector architecture figures; no measured data.

Run: MPLCONFIGDIR=/tmp/icra_mpl python3 experiments/icra_commissioning/thesis_plan.py
Outputs are explicitly conceptual/proposed, never experimental evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", "/tmp/icra_mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs/studies/icra_commissioning_20260905/thesis_plan"
INK = "#243340"
BLUE = "#e7eff7"
GREEN = "#e7f1ec"
AMBER = "#fff1d8"
GREY = "#f1f3f5"
EDGE = "#577083"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
                     "savefig.facecolor": "white"})

CAPTIONS = {
    "01_system": ("One commissioned model connects estimation and navigation",
        "Proposed system architecture. Commissioning supplies the current-observation model and the "
        "future-observation model. The existing detector, metric correction, robot dynamics, planner "
        "objective and controller are retained. Reference poses are offline labels and evaluation only; "
        "the new model is not yet validated online."),
    "02_perception": ("Perception produces a measurement of one declared robot reference",
        "Per-camera observation construction. The existing NN adds a metric correction to the raw "
        "ground projection of the bbox bottom centre. Covariance is calibrated for the corrected "
        "ground-position residual. Every opportunity retains its capture identity and outcome. "
        "Image features are optional; no perception posterior is treated as a fresh observation."),
    "03_fusion": ("Fusion must account for the age and reuse of camera evidence",
        "Proposed primary fusion comparison. Individual corrected camera readings update one robot "
        "filter at their capture times, with fixed robot Q. The existing robust network aggregation "
        "remains a baseline. Persistent error states and cross-camera covariance are added only if "
        "synchronized residuals justify them; those mechanisms are not established results."),
    "04_gp": ("The GP predicts future observation availability",
        "Proposed planner-facing interface. Reuse the spatial availability GP first; retain a separate "
        "conditional measurement-quality model. Future image features are marginalized through "
        "commissioned quality regimes. A GP over log covariance scale is optional. GP map uncertainty, "
        "measurement covariance R and robot covariance P are distinct. Joint outcomes and modeled "
        "temporal dependence enter the belief rollout."),
    "05_studies": ("Each study tests a different link in the scientific chain",
        "Execution and evidence dependencies. Grouped commissioning data establish the sensor model; "
        "identical-log replay tests fusion; independently collected fixed routes test prediction; "
        "matched closed-loop trials test navigation. A small estimator/planner factorial separates "
        "the two sources of improvement. The first runs size the final study; frames are not replicates."),
}


def base(title, subtitle="Proposed architecture | conceptual figure, not a measured result"):
    fig = plt.figure(figsize=(10.8, 7.4))
    fig.text(.045, .952, title, fontsize=16, weight="bold", color=INK, va="top")
    fig.text(.045, .911, subtitle, fontsize=9.5, color=EDGE, va="top")
    ax = fig.add_axes([.035, .19, .93, .68])
    ax.set(xlim=(0, 100), ylim=(0, 70)); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, label, color=GREY, dashed=False, size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
        linewidth=1.05, edgecolor=EDGE, facecolor=color, linestyle="--" if dashed else "-", zorder=2))
    ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=size, color=INK,
            linespacing=1.35, zorder=3)


def arrow(ax, points, label=None, label_pos=None, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0],b[0]],[a[1],b[1]], color=EDGE, lw=1.2,
                linestyle="--" if dashed else "-", zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2],points[-1], arrowstyle="-|>", mutation_scale=11,
        color=EDGE, linewidth=1.2, linestyle="--" if dashed else "-", zorder=1))
    if label:
        ax.text(*label_pos, label, fontsize=9, color=INK, ha="center", va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=4)


def note(ax,x,y,label):
    ax.text(x,y,label,fontsize=10,color=EDGE,ha="center",va="center",linespacing=1.4)


def footer(fig, caption, index):
    fig.text(.045,.136,"\n".join(textwrap.wrap(caption, 137)), fontsize=9.3,
             color=INK, va="top", linespacing=1.4)
    fig.text(.045,.028,"Commissioned camera information | thesis figure plan | 5 September 2026",
             fontsize=8.5,color=EDGE)
    fig.text(.956,.028,str(index),fontsize=8.5,color=EDGE,ha="right")


def diagram_system():
    fig, ax=base(CAPTIONS["01_system"][0])
    box(ax,3,56,26,11,"Commissioning records\nReference, images, misses",BLUE)
    box(ax,37,56,28,11,"Commissioned sensor model\nMean, R, usability, dependence",GREEN)
    arrow(ax,[(29,61.5),(37,61.5)])
    box(ax,3,33,26,12,"Current images\nExisting perception",BLUE)
    box(ax,37,33,28,12,"Robot estimator\nCurrent belief and history",GREEN)
    box(ax,73,33,24,12,"Future belief rollout\nPossible camera outcomes",GREEN)
    arrow(ax,[(51,56),(51,45)])
    arrow(ax,[(37,59),(32,59),(32,49),(22,49),(22,45)])
    arrow(ax,[(65,61.5),(85,61.5),(85,45)])
    arrow(ax,[(29,39),(37,39)])
    arrow(ax,[(65,39),(73,39)])
    box(ax,3,8,26,12,"Robot + measured odometry\nExisting controller",GREY)
    box(ax,37,8,28,12,"Existing planner\nSame objective and constraints",GREY)
    arrow(ax,[(51,33),(51,20)])
    arrow(ax,[(73,36),(69,36),(69,14),(65,14)],"Forecast",(70,24))
    arrow(ax,[(65,18),(71,18),(71,29),(90,29),(90,33)],"Candidate motions",(84,26))
    arrow(ax,[(37,14),(29,14)])
    arrow(ax,[(16,20),(16,33)])
    arrow(ax,[(29,19),(33,19),(33,34),(37,34)],"Odometry",(31,26))
    note(ax,83,10,"No future images\nNo runtime ground truth")
    return fig


def diagram_perception():
    fig, ax=base(CAPTIONS["02_perception"][0])
    box(ax,2,55,25,12,"Image + capture identity\nOriginal pixel coordinates",BLUE)
    box(ax,36,55,27,12,"Frozen YOLO\nBBox + confidence",GREY)
    box(ax,73,55,25,12,"Missed detection\nLog; no camera update",AMBER)
    arrow(ax,[(27,61),(36,61)])
    arrow(ax,[(63,61),(73,61)],"Miss",(68,64))
    box(ax,2,34,25,12,"Calibrated floor projection\nRaw bbox bottom centre",GREY)
    box(ax,36,34,27,12,"Existing metric NN\nCorrect robot-reference XY",GREY)
    box(ax,73,34,25,12,"Observable features\nGeometry, bbox, score\nOptional RGB features",BLUE,size=10)
    arrow(ax,[(42,55),(42,50),(14.5,50),(14.5,46)],"Hit",(28,50))
    arrow(ax,[(27,40),(36,40)])
    arrow(ax,[(49.5,55),(49.5,46)])
    arrow(ax,[(58,55),(58,51),(85.5,51),(85.5,46)])
    box(ax,2,11,25,13,"Per-camera observation\nz [m], R [m²], h = XY\nTime, ID, status, reason",GREEN,size=10)
    box(ax,36,11,27,13,"Frozen residual mean offset\nApply once to corrected z",GREEN)
    box(ax,73,11,25,13,"Commissioned R and support\nFor corrected XY residuals",GREEN,size=10)
    arrow(ax,[(49.5,34),(49.5,24)])
    arrow(ax,[(85.5,34),(85.5,24)])
    arrow(ax,[(36,18),(27,18)])
    arrow(ax,[(85.5,11),(85.5,5),(14.5,5),(14.5,11)])
    arrow(ax,[(8,55),(0,55),(0,18),(2,18)],dashed=True)
    return fig


def diagram_fusion():
    fig, ax=base(CAPTIONS["03_fusion"][0])
    box(ax,2,55,27,12,"Measured odometry\nTimestamped motion inputs",BLUE)
    box(ax,37,55,27,12,"Robot prediction\nExisting dynamics + fixed Q",GREY)
    box(ax,73,55,25,12,"Optional camera-error states\nFitted persistence model",AMBER,dashed=True,size=10)
    arrow(ax,[(29,61),(37,61)])
    arrow(ax,[(73,61),(64,61)],dashed=True)
    box(ax,2,31,27,13,"Individual camera readings\nSame reference and units\nPer-camera capture time",BLUE)
    box(ax,37,31,27,13,"Time and identity handling\nEach frame contributes once\nCapture-time update + replay",GREEN,size=10)
    box(ax,73,31,25,13,"Central robot filter\nStable update; Joseph form",GREEN,size=10)
    arrow(ax,[(29,37.5),(37,37.5)])
    arrow(ax,[(64,37.5),(73,37.5)])
    arrow(ax,[(50.5,55),(50.5,44)])
    arrow(ax,[(85.5,55),(85.5,44)],dashed=True)
    box(ax,37,6,27,13,"Audit record\nPre-gate innovation and S\nAccepted, rejected, dropped",BLUE,size=10)
    box(ax,73,6,25,13,"Robot posterior\nPose and covariance P",GREEN)
    arrow(ax,[(85.5,31),(85.5,19)])
    arrow(ax,[(73,12.5),(64,12.5)])
    arrow(ax,[(98,12.5),(100,12.5),(100,69),(50.5,69),(50.5,67)])
    note(ax,15,14,"Start with block-diagonal R.\nTest synchronized residuals.\nRetain network fusion baseline.")
    return fig


def diagram_gp():
    fig, ax=base(CAPTIONS["04_gp"][0])
    box(ax,2,54,27,13,"Future configuration s\nPosition, heading if supported\nCamera geometry + known map",BLUE,size=10)
    box(ax,37,54,27,13,"Availability model q(s)\nReuse spatial GP candidate\nTrain on hits AND misses",GREEN,size=10)
    box(ax,73,54,25,13,"Map-model uncertainty\nSeparate from R and P",AMBER,size=10)
    arrow(ax,[(29,60.5),(37,60.5)])
    arrow(ax,[(64,60.5),(73,60.5)])
    box(ax,2,29,27,14,"Conditional quality model\nGeometry / learned regimes\nOptional GP log-scale later",GREEN,size=10)
    box(ax,37,29,27,14,"Joint future outcomes\nCamera misses + quality\nIntegrate unknown image features",GREEN,size=9.8)
    box(ax,73,29,25,14,"Belief rollout\nCurrent P + motion + cadence\nModeled error persistence",GREY,size=10)
    arrow(ax,[(15.5,54),(15.5,43)])
    arrow(ax,[(50.5,54),(50.5,43)])
    arrow(ax,[(29,36),(37,36)])
    arrow(ax,[(64,36),(73,36)])
    arrow(ax,[(85.5,54),(85.5,43)],dashed=True)
    box(ax,37,4,27,13,"Expected posterior quality\nAverage hit / miss branches\nValidate on held-out routes",GREEN,size=10)
    box(ax,73,4,25,13,"Existing planning objective\nCompare candidate motions",GREY,size=10)
    arrow(ax,[(79,29),(79,23),(50.5,23),(50.5,17)])
    arrow(ax,[(64,10.5),(73,10.5)])
    note(ax,15,12,"R is sensor scatter.\nP is robot uncertainty.\nA GP is a model of a function.")
    return fig


def diagram_studies():
    fig, ax=base(CAPTIONS["05_studies"][0],"Planned evidence sequence | no outcome or performance gain is assumed")
    rows=[(55,"A  Measurement model","Grouped poses +\nnuisance variation","Bias, R and support\nCommissioning budget"),
          (38,"B  Multi-camera fusion","Identical synchronized logs\nand measured odometry","Accuracy, coverage, sharpness\nand dependence"),
          (21,"C  Future prediction","Independent executions\nof fixed routes","Forecast magnitude\nand route ranking"),
          (4,"D  Navigation","Matched conditions\nEstimator × planner","Success and failures\nTravel time and path cost")]
    for y,title,data,result in rows:
        box(ax,2,y,25,12,title,GREEN,size=11)
        box(ax,34,y,29,12,data,BLUE,size=9.7)
        box(ax,70,y,28,12,result,GREY,size=9.7)
        arrow(ax,[(27,y+6),(34,y+6)])
        arrow(ax,[(63,y+6),(70,y+6)])
    for y1,y2 in [(55,50),(38,33),(21,16)]:
        arrow(ax,[(14.5,y1),(14.5,y2)])
    return fig


def textpage(title, subtitle, sections, page):
    fig=plt.figure(figsize=(10.8,7.4))
    fig.text(.055,.945,title,fontsize=19,weight="bold",color=INK,va="top")
    fig.text(.055,.893,subtitle,fontsize=10,color=EDGE,va="top")
    y=.84
    for heading,paragraph in sections:
        fig.text(.055,y,heading,fontsize=12,weight="bold",color=INK,va="top")
        y-=.034
        lines=textwrap.wrap(paragraph,125)
        fig.text(.055,y,"\n".join(lines),fontsize=10.5,color=INK,va="top",linespacing=1.4)
        y-=len(lines)*.027+.022
    if y<.07:
        raise RuntimeError(f"Text page overflow: {title}, {y}")
    fig.text(.055,.03,"Planning document | 5 September 2026 | Existing evidence remains development-only",
             fontsize=8.5,color=EDGE)
    fig.text(.955,.03,str(page),fontsize=9,color=EDGE,ha="right")
    return fig


PLAN_PAGES=[
    ("Research plan: predict useful camera information", "Scope: framing, experiments and thesis figures. Manuscript development remains paused.", [
        ("Research question", "What must commissioning learn for a fixed-camera network to predict its localization "
         "value during estimation and along a future route? Accuracy, calibration and navigation utility are separate "
         "claims. The method is evaluated in this camera installation; broader transfer must be tested separately."),
        ("Keep the working components", "Retain the frozen YOLO, existing metric-reference correction NN, calibrated "
         "camera geometry, robot process model and Q, planner objective and controller. The camera output is ground "
         "reference XY in metres. The original Q-identification artifact still needs verification; freezing Q is a control, "
         "not proof that Q is correct."),
        ("Change the measurement interface first", "Calibrate covariance for corrected ground-position residuals. Apply "
         "the same learned mean offset in every covariance arm. Forward fresh per-camera evidence to one robot filter; "
         "retain existing robust network fusion as a baseline. Log all opportunities and exact update outcomes."),
        ("Give the GP a precise role", "Use the existing spatial GP as an availability candidate, q_i(s) = probability of a "
         "usable observation. Keep conditional hit covariance separate. Only add a GP for residual covariance scale if "
         "constant and geometry-conditioned models leave a reproducible spatial gap."),
        ("Current boundary", "Calibration, fixed-Q replay and a future-outcome diagnostic exist offline. New covariance "
         "and future models are not yet installed online. The existing figure package is development evidence, not "
         "replicated fusion or navigation validation. These diagrams show the proposed architecture.")]),
    ("Execution sequence and advancement criteria", "Freeze final evaluation before model selection; retain failures and ordinary conditions.", [
        ("1. Contract and data collection", "Add complete per-camera opportunity logging and crop references; verify "
         "live/replay mean and covariance agreement, timestamps, frame deduplication, delay handling and measured "
         "odometry. Freeze scenario families: clear complementary views, transitions, partial occlusion and dropout. "
         "Pilot independent executions, then size and freeze the final manifest from run-level variability."),
        ("2. Commission and freeze the measurement model", "Use separate grouped mean-training, covariance-fit, selection "
         "and final-test roles. Compare full constant, geometry, and selected contextual covariance with identical "
         "mean and acceptance. Advance complexity only for held-out gains in score/calibration at useful sharpness. "
         "Keep constant covariance if it performs equally well."),
        ("3. Establish sequential fusion", "Replay identical logs for each camera and their combination. Keep Q, "
         "initialization, rate and acceptance fixed. Measure reference residual dependence within runs and synchronized "
         "camera pairs. If independence fails materially, fit a small persistence/shared-error model on development "
         "runs and test it on unseen runs. Do not select it from the final trajectories."),
        ("4. Validate future prediction", "Compare constant, geometry, empirical local-outcome and GP availability models "
         "with the same conditional quality model. Integrate possible hits/misses, camera combinations and supported "
         "quality regimes. Match runtime cadence and dependence. Advance to navigation after held-out fixed routes "
         "test uncertainty magnitude, forecast calibration and route ordering."),
        ("5. Evaluate navigation", "Use a small 2 by 2 comparison: baseline/selected estimator crossed with baseline/selected "
         "future model. Match starts, goals, environments and nuisance seeds; keep planner/controller settings fixed. "
         "Report run-level position and heading error, failures, success, path/time cost, correction gaps and latency.")]),
    ("GP decision: availability first, quality only if needed", "Reuse the repository GP machinery; do not equate its posterior variance with sensor noise.", [
        ("What is implemented", "ObservabilityGP aggregates binary events on a spatial grid, applies Beta smoothing, "
         "and uses the canonical latent RBF GP in logit space. Its wrapper currently returns sigmoid(latent mean) and "
         "discards latent standard deviation. This is an approximate probability model, not a fitted Bernoulli GP classifier. "
         "The planner already accepts gridded probability artifacts."),
        ("Required adapter work", "Fit per-camera maps with grouped splits, explicit all-opportunity labels and sample "
         "counts. Validate the approximation against geometry/local frequency and, if needed, a Bernoulli/binomial GP. "
         "Retain map-model uncertainty; integrate the link over latent uncertainty when making posterior-predictive "
         "claims. Existing exports set conservative probability equal to mean and use a placeholder latent spread."),
        ("Inputs and support", "Begin with position and observable camera geometry. Add periodic heading features only "
         "when supported. Choose whether training inputs represent surveyed configuration or estimated position; do "
         "not silently mix them. Account for uncertain future position. Align numerical and optimizer support rules: "
         "the present numerical adapter rejects out-of-grid queries while its CasADi path clamps them."),
        ("Optional covariance GP", "Use a positive scale R_i(s) = exp(g_i(s)) R_geometry,i(s) as the first covariance-GP "
         "candidate. Fit to frozen-mean residual likelihood, not GP prediction variance. A scalar scale preserves "
         "ellipse shape; use a Cholesky model only if held-out directional errors justify it. Unknown future image "
         "features require a distribution over quality regimes, not an invented future confidence score."),
        ("Limits and sources", "A static GP kernel is not by itself a sequential measurement-noise model. Deterministic "
         "renders do not become independent Bernoulli trials: probability must represent declared state/nuisance variation. "
         "Reference: Rasmussen and Williams, Gaussian Processes for Machine Learning, chapters 2 and 3 (2006), "
         "gaussianprocess.org/gpml. Exact repository paths and figure captions are in the accompanying plan index.")]),
]

# Keep the implementation caveats legible instead of shrinking the GP page.
gp_title, gp_subtitle, gp_sections = PLAN_PAGES.pop()
PLAN_PAGES.append((gp_title, gp_subtitle, gp_sections[:3]))
PLAN_PAGES.append(("Choose GP complexity from predictive evidence",
    "The commissioned observation model is the contribution candidate; the GP is one implementation.",
    gp_sections[3:] + [
        ("Selection rule", "For availability, compare constant rate, geometry, local empirical outcomes and GP on "
         "the same held-out opportunities. Evaluate Brier score, calibration and supported coverage; then test "
         "future belief forecasts and route ordering. If the GP adds no useful predictive value, retain the simpler "
         "model and report that result. Do not tune a conservative probability adjustment on final navigation runs."),
        ("Result figures to earn", "Measurement: raw/corrected bias fields, directional error ellipses, calibration "
         "and commissioning-budget curves. Fusion: individual/fused estimates during complementarity and dropout, "
         "plus within-run temporal dependence. Prediction: availability maps and predicted versus realized route "
         "uncertainty. Navigation: matched trajectories, uncertainty, failures and travel cost. Every empirical "
         "panel will state its manifest, sample population and independent evaluation units.")]))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    outputs=[]
    pack=OUT/"camera_information_plan.pdf"
    with PdfPages(pack,metadata={"Title":"Commissioned camera information: execution plan and architecture",
                 "Author":"UnembodiedNavigation research project", "Subject":"Proposed architecture; no measured results"}) as pdf:
        for index,(title,subtitle,sections) in enumerate(PLAN_PAGES,1):
            fig=textpage(title,subtitle,sections,index); pdf.savefig(fig); plt.close(fig)
        for index,(key,fn) in enumerate(zip(CAPTIONS,[diagram_system,diagram_perception,diagram_fusion,diagram_gp,diagram_studies]),len(PLAN_PAGES)+1):
            fig=fn(); footer(fig,CAPTIONS[key][1],index); pdf.savefig(fig)
            for ext in ("pdf","svg","png"):
                dest=OUT/f"{key}.{ext}"
                # Thesis inserts contain just the diagram; captions are separate.
                bounds=fig.axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                fig.savefig(dest,dpi=300,bbox_inches=bounds.expanded(1.035,1.08))
                outputs.append(dest)
            plt.close(fig)
    outputs.append(pack)
    index_lines=["# Thesis architecture figures", "", "These are conceptual figures of the proposed architecture. "
                 "They contain no synthetic results or measured performance claims.", "",
                 "Current plan and status: `docs/ICRA_STATUS.md`.", "",
                 "Each figure is supplied as a tightly cropped vector PDF, editable SVG and 300 dpi PNG. "
                 "The combined PDF includes the execution plan and all figures.", ""]
    for key,(title,caption) in CAPTIONS.items():
        index_lines.extend([f"## {key}: {title}", "", caption, "",
            f"[Vector PDF]({key}.pdf) | [Editable SVG]({key}.svg) | [300 dpi PNG]({key}.png)", "",
            f"LaTeX: `\\includegraphics[width=\\linewidth]{{{key}.pdf}}`", ""])
    (OUT/"FIGURE_INDEX.md").write_text("\n".join(index_lines))
    manifest={"kind":"conceptual_architecture_and_execution_plan", "contains_measured_results":False,
        "source":str(Path(__file__).relative_to(ROOT)),
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "files":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs},
        "sources":["src/reliability/reliability/learned_box_correction.py",
            "src/reliability/reliability/observation_gp.py",
            "scripts/visibility_comparison/fit_belief_aware_gp.py",
            "src/reliability/reliability/observation_planner_artifact.py",
            "src/planning/planning/core/visibility_gp_map.py",
            "experiments/icra_commissioning/README.md"],
        "literature":["https://gaussianprocess.org/gpml/chapters/RW2.pdf",
                      "https://gaussianprocess.org/gpml/chapters/RW3.pdf"]}
    manifest["source_file_sha256"]={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                    for name in manifest["sources"]}
    (OUT/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(pack)


if __name__ == "__main__":
    main()
