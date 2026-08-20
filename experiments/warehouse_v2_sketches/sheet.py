#!/usr/bin/env python3
"""Contact sheet of the grabbed frames next to the sketch's expectation."""
import pathlib, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent
import sys as _sys
_sys.path.insert(0, str(HERE))
from warehouse_v2 import build as _build           # noqa: E402

_L = _build()
LABEL = {c.name: f"{c.name}  ({c.x:+.2f}, {c.y:+.2f}, {c.z:.1f} m)  yaw {c.yaw_deg:+.0f}  "
                 f"pitch {c.pitch_deg:.0f}   {c.mount}" for c in _L.cameras}
LABEL["TOP"] = "TOP  presentation overview, 19 m, 90 deg, north up"
LABEL["PLAN"] = "PLAN  checking camera, 42 m, 0.90 rad -- near-orthographic, north up"


def main(tag="iter", src="frames"):
    d = HERE / src
    order = ["PLAN", "TOP", "A", "B", "C", "D", "E"]
    imgs = []
    for k in order:
        p = d / f"{tag}_{k}.png"
        if p.exists():
            imgs.append((k, Image.open(p).convert("RGB")))
    if not imgs:
        print("no frames"); return 1
    w = 900
    tiles = []
    for k, im in imgs:
        im = im.resize((w, int(im.height * w / im.width)))
        t = Image.new("RGB", (w, im.height + 26), "#12181d")
        t.paste(im, (0, 26))
        dr = ImageDraw.Draw(t)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
        except Exception:
            f = ImageFont.load_default()
        dr.text((8, 5), LABEL.get(k, k), fill="#e8eef3", font=f)
        tiles.append(t)
    # per-row heights: the plan camera is 4:3 and the wall cameras 16:9, so one
    # global tile height would leave a black band under every wall camera
    cols = 2
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    row_h = [max(t.height for t in r) for r in rows]
    sheet = Image.new("RGB", (cols * w + 12, sum(row_h) + 12 * len(rows)), "#12181d")
    y = 0
    for r, hh in zip(rows, row_h):
        for j, t in enumerate(r):
            sheet.paste(t, (j * (w + 12), y))
        y += hh + 12
    o = HERE / "figures" / f"gazebo_{tag}.png"
    sheet.save(o)
    print(f"wrote {o}  ({sheet.width}x{sheet.height})")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
