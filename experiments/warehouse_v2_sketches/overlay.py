#!/usr/bin/env python3
"""Draw the PLANNED geometry on top of the rendered plan view.

The plan camera looks straight down from 42 m, so world -> pixel is a plain
pinhole divide. Verified against two independent features of the render: the
inner face of the south wall (y = -10) lands at v = 994 px and the inner face of
the east wall (x = +12) at u = 1273 px, both within a couple of pixels of where
the render puts them.

Anything drawn here that does not sit on the thing it describes is a real
mismatch between the layout and the world, not a perspective artefact.
"""
from __future__ import annotations

import math
import pathlib
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from warehouse_v2 import SITE, build          # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
CAM_Z, HFOV, W, H = 42.0, 0.90, 1600, 1200
F = (W / 2) / math.tan(HFOV / 2)


def uv(x, y, z=0.0):
    d = CAM_Z - z
    return W / 2 + F * x / d, H / 2 - F * y / d


def rect(dr, xmin, xmax, ymin, ymax, colour, width=2, z=0.0):
    p = [uv(xmin, ymin, z), uv(xmax, ymin, z), uv(xmax, ymax, z), uv(xmin, ymax, z)]
    dr.line(p + [p[0]], fill=colour, width=width)


def main(tag="it04"):
    src = HERE / "frames" / f"{tag}_PLAN.png"
    if not src.exists():
        print(f"no {src}"); return 1
    im = Image.open(src).convert("RGB")
    dr = ImageDraw.Draw(im)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 17)
        fs = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except Exception:
        f = fs = ImageFont.load_default()

    L = build()
    rect(dr, -12, 12, -10, 10, "#ff00ff", 2)                       # inner wall faces
    rect(dr, *SITE[:2], *SITE[2:], "#00b0ff", 2)                   # site boundary
    for z in L.zones:
        rect(dr, z.xmin, z.xmax, z.ymin, z.ymax, "#00ff88", 3)
        u, v = uv(z.cx, z.cy)
        dr.text((u - 16, v - 8), z.name.split("_")[0], fill="#00ff88", font=fs)
    for c in L.cameras:
        u, v = uv(c.x, c.y)
        dr.ellipse([u - 9, v - 9, u + 9, v + 9], fill="#ff3355", outline="#ffffff", width=2)
        dr.text((u - 5, v - 9), c.name, fill="#ffffff", font=fs)
        yaw = math.radians(c.yaw_deg)
        u2, v2 = uv(c.x + 3.2 * math.cos(yaw), c.y + 3.2 * math.sin(yaw))
        dr.line([(u, v), (u2, v2)], fill="#ff3355", width=3)
    for d in (-8.5, -1.5, 5.5):                                    # dock doors
        u1, v1 = uv(d - 1.6, -10.0); u2, v2 = uv(d + 1.6, -10.0)
        dr.line([(u1, v1), (u2, v2)], fill="#ffd400", width=6)
    dr.text((20, 16), f"PLANNED geometry drawn over the {tag} render — "
                      f"magenta walls, blue site bound, green zones, yellow dock doors",
            fill="#ffffff", font=f)
    o = HERE / "figures" / f"overlay_{tag}.png"
    im.save(o)
    print(f"wrote {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
