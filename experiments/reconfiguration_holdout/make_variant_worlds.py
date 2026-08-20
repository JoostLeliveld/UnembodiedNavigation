#!/usr/bin/env python3
"""Generate changed-warehouse worlds from the frozen flagship world.

The five environments this study needs, all with identical camera calibration,
identical driveable lanes and the identical detector:

| key | layout | lighting | source |
|---|---|---|---|
| `L0`      | nominal    | nominal | the frozen flagship world, untouched |
| `L1`      | restocked  | nominal | generated here |
| `L2`      | independently restocked | nominal | generated here |
| `L0_lit`  | nominal    | changed | generated here |
| `L1_lit`  | restocked  | changed | generated here |

The flagship world is frozen evaluation infrastructure and is **read, never
written**.  Each variant is a derived file: same physics, same models, same camera
includes and poses, with exactly one or two edits applied to the text.

**Layout change** — added load on top of the rack segments chosen by
``choose_layout.py`` (L1) or the outcome-blind ``choose_second_layout.py`` (L2).
The stock boxes are emitted as links *inside* the existing
``warehouse_rack_occluders`` model rather than as a new model, so that every
consumer that already parses that model name — the capture's own occlusion oracle,
``unav_common.occlusion_geometry``, the dynamic-world oracle — sees the restock
automatically and no downstream tool can silently score the reconfigured world
against nominal geometry.  Each box carries both a collision and a visual, because
the renderer and the ray-cast oracle read different tags and a camera does not care
whether a load has physics.

**Lighting change** — the four named lights are replaced.  Two overhead lamps go
out, the third dims, and the directional light drops to a low angle at higher
intensity from a different bearing.  That is a lighting failure plus low-sun glare
through the dock: it changes image brightness, contrast and the entire shadow
structure while leaving every millimetre of geometry alone.  Nothing about the
scene the cameras look at moves, which is what makes it a clean appearance-only
condition.

    python3 experiments/reconfiguration_holdout/make_variant_worlds.py

Writes the four derived SDF files into src/sim/gazebo_worlds/worlds/ and copies them into
install/sim/share/sim/gazebo_worlds/worlds/ (campaigns and the capture's world
resolver load from install, not src), plus a study-local world_profiles.yaml
carrying one profile entry per variant.  The shared
src/experiments/config/world_profiles.yaml is not edited.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import sys

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts/shared"))
from paths import repo_root  # noqa: E402

REPO = repo_root(HERE)
SRC_WORLDS = REPO / "src/sim/gazebo_worlds/worlds"
INSTALL_WORLDS = REPO / "install/sim/share/sim/gazebo_worlds/worlds"
BASE_WORLD_FILE = "warehouse_full_4cam.world.sdf"
BASE_WORLD_NAME = "warehouse_full_4cam"
SHARED_PROFILE = REPO / "src/experiments/config/world_profiles.yaml"
LAYOUT_L1 = REPO / "logs/studies/reconfiguration_holdout/layout/layout_selected.json"
LAYOUT_L2 = HERE / "layouts/L2_layout.json"
STUDY_PROFILE = HERE / "world_profiles_variants.yaml"

GENERATED_BY = "experiments/reconfiguration_holdout/make_variant_worlds.py"

#: The changed-lighting condition, as the exact SDF the variant worlds carry.
#: Written out in full rather than as a set of deltas so the condition can be read
#: off the file: two lamps out, one dimmed to 0.35, and a low-angle 1.45-intensity
#: directional light from the west instead of a 0.74 overhead one.
LIGHTS_CHANGED = """    <light name="sun" type="directional"><pose>0 0 12 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>1.45</intensity><direction>-0.92 0.12 -0.37</direction></light>
    <light name="center_overhead_light" type="point"><pose>0 0.5 5.2 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>0.35</intensity><attenuation><range>18</range><constant>0.45</constant><linear>0.06</linear><quadratic>0.010</quadratic></attenuation></light>"""

LIGHT_LINE = re.compile(r'^\s*<light name="[^"]+".*?</light>\s*$', re.M)


def stock_links(segments: list[dict]) -> str:
    """Rack-top load, as SDF links carrying a collision and a visual box each."""
    parts = []
    for seg in segments:
        name = seg["name"]
        h = float(seg["stock_height_m"])
        sx = float(seg["xmax"]) - float(seg["xmin"])
        sy = float(seg["ymax"]) - float(seg["ymin"])
        cx = 0.5 * (float(seg["xmin"]) + float(seg["xmax"]))
        cy = 0.5 * (float(seg["ymin"]) + float(seg["ymax"]))
        cz = float(seg["top_z"]) + h / 2.0
        geom = f"<geometry><box><size>{sx:.3f} {sy:.3f} {h:.3f}</size></box></geometry>"
        colour = "0.72 0.55 0.30 1"
        parts.append(
            f'      <link name="stock_{name}"><pose>{cx:.3f} {cy:.3f} {cz:.3f} 0 0 0</pose>'
            f'<collision name="collision">{geom}</collision>'
            f'<visual name="visual">{geom}'
            f'<material><ambient>{colour}</ambient><diffuse>{colour}</diffuse></material>'
            f'</visual></link>'
        )
    return "\n".join(parts)


def apply_restock(text: str, segments: list[dict]) -> str:
    anchor = '<model name="warehouse_rack_occluders"><static>true</static>\n'
    if anchor not in text:
        raise RuntimeError("rack occluder model anchor not found in the base world")
    banner = (
        f"      <!-- Rack restock added by {GENERATED_BY}.  "
        f"{len(segments)} of the warehouse's rack segments carry one extra layer of "
        f"stock. Edit the generator, not this file. -->\n"
    )
    return text.replace(anchor, anchor + banner + stock_links(segments) + "\n", 1)


def apply_lighting(text: str) -> str:
    matches = LIGHT_LINE.findall(text)
    if len(matches) != 4:
        raise RuntimeError(f"expected 4 <light> elements in the base world, found {len(matches)}")
    banner = (
        f"    <!-- Changed lighting applied by {GENERATED_BY}: two overhead lamps out,\n"
        f"         the third dimmed, and a low-angle high-intensity directional light\n"
        f"         from the west. Geometry is untouched. -->\n"
    )
    text = LIGHT_LINE.sub("", text, count=4)
    return text.replace("  </world>", banner + LIGHTS_CHANGED + "\n  </world>", 1)


def variant_text(base: str, world_name: str, *, restock: list[dict] | None,
                 lighting: bool) -> str:
    text = base.replace(f'<world name="{BASE_WORLD_NAME}">',
                        f'<world name="{world_name}">', 1)
    header = (
        f"<!-- GENERATED by {GENERATED_BY} from {BASE_WORLD_FILE}.\n"
        f"     Derived variant: layout={'restocked' if restock else 'nominal'}, "
        f"lighting={'changed' if lighting else 'nominal'}.\n"
        f"     The flagship world is frozen evaluation infrastructure and is never"
        f" written by this tool. -->\n"
    )
    text = text.replace("<sdf ", header + "<sdf ", 1)
    if restock:
        text = apply_restock(text, restock)
    if lighting:
        text = apply_lighting(text)
    return text


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--layout",
        "--layout-l1",
        dest="layout_l1",
        default=str(LAYOUT_L1),
        help="frozen L1 geometry-selected layout (legacy --layout alias retained)",
    )
    ap.add_argument(
        "--layout-l2",
        default=str(LAYOUT_L2),
        help="frozen outcome-blind second-reconfiguration layout",
    )
    ap.add_argument("--no-install-copy", action="store_true")
    args = ap.parse_args(argv)

    base_path = SRC_WORLDS / BASE_WORLD_FILE
    base = base_path.read_text(encoding="utf-8")
    layout_l1 = json.loads(Path(args.layout_l1).read_text(encoding="utf-8"))
    layout_l2 = json.loads(Path(args.layout_l2).read_text(encoding="utf-8"))
    segments_l1 = layout_l1["restocked_segments"]
    segments_l2 = layout_l2["restocked_segments"]
    if layout_l2.get("environment_key") != "L2":
        raise ValueError("--layout-l2 is not an L2 layout artifact")
    if layout_l2.get("selection", {}).get("outcome_blind") is not True:
        raise ValueError("--layout-l2 does not declare outcome-blind selection")
    print(f"[worlds] base {base_path.relative_to(REPO)} ({len(base)} chars); "
          f"L1 restock = {len(segments_l1)} segments at "
          f"+{layout_l1['stock_height_m']} m; L2 restock = {len(segments_l2)} "
          f"segments at +{layout_l2['stock_height_m']} m")

    variants = {
        "L1": ("warehouse_full_4cam_recfg", segments_l1, False, "L1 restock"),
        "L2": ("warehouse_full_4cam_recfg2", segments_l2, False, "L2 independent restock"),
        "L0_lit": ("warehouse_full_4cam_lit", None, True, "nominal"),
        "L1_lit": ("warehouse_full_4cam_recfg_lit", segments_l1, True, "L1 restock"),
    }

    written = {}
    layout_labels = {}
    for key, (world_name, restock, lighting, layout_label) in variants.items():
        text = variant_text(base, world_name, restock=restock, lighting=lighting)
        path = SRC_WORLDS / f"{world_name}.world.sdf"
        path.write_text(text, encoding="utf-8")
        written[key] = world_name
        layout_labels[key] = layout_label
        n_stock = text.count('<link name="stock_')
        n_light = len(LIGHT_LINE.findall(text))
        print(f"[worlds] {key:7s} -> {path.name:38s} "
              f"stock links {n_stock:2d}, lights {n_light}")
        if not args.no_install_copy:
            INSTALL_WORLDS.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, INSTALL_WORLDS / path.name)

    # One profile entry per variant: a deep copy of the flagship's own entry, so
    # camera intrinsics, spawn pose, visibility bounds and every declared lane are
    # identical by construction and cannot drift from the nominal environment.
    shared = yaml.safe_load(SHARED_PROFILE.read_text(encoding="utf-8"))
    base_entry = shared["worlds"][BASE_WORLD_FILE]
    out = {"camera_intrinsics": copy.deepcopy(shared["camera_intrinsics"]),
           "worlds": {BASE_WORLD_FILE: copy.deepcopy(base_entry)}}
    for key, world_name in written.items():
        entry = copy.deepcopy(base_entry)
        entry["world_name"] = world_name
        entry["investigation_focus"] = (
            f"Reconfiguration-holdout variant {key} of {BASE_WORLD_NAME}: "
            f"layout={layout_labels[key]}, "
            f"lighting={'changed' if key.endswith('lit') else 'nominal'}. "
            f"Generated by {GENERATED_BY}; camera calibration, spawn pose and all "
            f"declared lanes are copied verbatim from the flagship entry."
        )
        out["worlds"][f"{world_name}.world.sdf"] = entry
    STUDY_PROFILE.write_text(
        "# GENERATED by " + GENERATED_BY + " -- do not hand-edit.\n"
        "# Variant profiles for the reconfiguration holdout. The shared\n"
        "# src/experiments/config/world_profiles.yaml is deliberately left alone.\n"
        + yaml.safe_dump(out, sort_keys=False, width=200),
        encoding="utf-8",
    )
    print(f"[worlds] wrote {STUDY_PROFILE.relative_to(REPO)} with "
          f"{len(out['worlds'])} world entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
