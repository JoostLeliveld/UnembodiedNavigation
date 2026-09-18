#!/usr/bin/env python3
"""Derive the development timing world from the CANONICAL world.

The fast world exists only to make development runs cheaper. It must therefore
differ from `warehouse_v2.world.sdf` in exactly one way that cannot reach a
rendered pixel: contact sensors at 20 Hz instead of 60 Hz. Contact rate touches
the physics contact report and nothing the cameras see.

Deriving it rather than generating it independently is the point. `make_world.py
--fast` builds from the layout description, and that description no longer
reproduces the canonical world: it emits no `obs_Cm_p3` and the pre-2026-09-18
camera C pitch. A fast world built that way silently diverges in geometry while
claiming to be a twin, which is exactly the defect that retired
`warehouse_v2_low_cpu.world.sdf`.

So: read the canonical world, substitute the rate, write the result. Geometry,
cameras, lighting and shadows then match by CONSTRUCTION rather than by claim.

Run this whenever `warehouse_v2.world.sdf` changes. `--check` verifies the
derived world is still current without writing, which is what CI should call.

    python3 experiments/warehouse_v2_sketches/derive_fast_world.py
    python3 experiments/warehouse_v2_sketches/derive_fast_world.py --check
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CANONICAL = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf'
DERIVED = REPO / 'src/sim/gazebo_worlds/worlds/warehouse_v2_fast.world.sdf'

CONTACT_FROM = '<update_rate>60</update_rate>'
CONTACT_TO = '<update_rate>20</update_rate>'

HEADER = """  <!-- DEVELOPMENT DERIVATIVE of warehouse_v2.world.sdf, for TIMING ONLY.

       Derived from the canonical world by ONE substitution: contact sensors
       60 Hz to 20 Hz. Nothing else differs, so the geometry, cameras, lighting
       and shadows match the canonical world by construction rather than by
       claim. Contact rate touches the physics contact report and never a
       rendered pixel, so images here are identical to the evidence world.

       Regenerate with experiments/warehouse_v2_sketches/derive_fast_world.py
       whenever warehouse_v2.world.sdf changes. Do NOT hand-edit, and do NOT
       use make_world.py with the fast flag: that generator cannot reproduce
       the canonical world (it emits no obs_Cm_p3 and the pre-2026-09-18 camera
       C pitch), so it would silently reintroduce a geometric divergence.

       Never capture evidence in this world.
  -->
"""


def derive(canonical_text: str) -> str:
    """Canonical world with the contact rate lowered and our header in front."""
    if CONTACT_FROM not in canonical_text:
        raise SystemExit(f'canonical world has no {CONTACT_FROM}; the rate may have moved')
    fast = canonical_text.replace(CONTACT_FROM, CONTACT_TO)
    match = re.search(r'<sdf version="[^"]+">\n', fast)
    if not match:
        raise SystemExit('canonical world has no <sdf> opening tag')
    body = re.sub(r'^\s*<!--.*?-->\n', '', fast[match.end():], count=1, flags=re.S)
    return fast[:match.end()] + HEADER + body


def core(text: str) -> str:
    """Comment-free, rate-normalised body, for comparing the two worlds."""
    return re.sub(r'<!--.*?-->', '', text, flags=re.S).replace(CONTACT_TO, CONTACT_FROM)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--check', action='store_true',
                    help='verify the derived world is current; write nothing')
    args = ap.parse_args()

    canonical = CANONICAL.read_text()
    expected = derive(canonical)

    if args.check:
        if not DERIVED.is_file():
            print(f'MISSING: {DERIVED}')
            return 1
        actual = DERIVED.read_text()
        if core(actual) != core(canonical):
            print('STALE: the fast world no longer matches the canonical geometry.')
            print('       Re-run without --check.')
            return 1
        if actual != expected:
            print('STALE: the fast world differs from what derivation produces '
                  '(header drift).')
            return 1
        n = actual.count(CONTACT_TO)
        print(f'current: fast world matches canonical geometry, {n} contact sensors at 20 Hz')
        return 0

    DERIVED.write_text(expected)
    print(f'wrote {DERIVED.relative_to(REPO)}')
    print(f'  {expected.count(CONTACT_TO)} contact sensors lowered 60 -> 20 Hz')
    print('  geometry, cameras and lighting identical to the canonical world')
    return 0


if __name__ == '__main__':
    sys.exit(main())
