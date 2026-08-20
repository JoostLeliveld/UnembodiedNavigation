#!/usr/bin/env python3
"""Sketch beside the built world, in both stock states: does it resemble the plan?"""
from __future__ import annotations
import pathlib, sys
from PIL import Image, ImageDraw, ImageFont

HERE = pathlib.Path(__file__).resolve().parent


def crop_hall(p, box=(190, 85, 1410, 1105)):
    """Trim the plan render to the building, so the three panels are the same subject."""
    return Image.open(p).convert("RGB").crop(box)


def main():
    sketch = Image.open(HERE / "figures/warehouse_v2.png").convert("RGB")
    # left panel of the sketch sheet is the plan
    sketch = sketch.crop((60, 150, 1250, 1330))
    peak = crop_hall(HERE / "frames/it10_PLAN.png")
    ship = crop_hall(HERE / "frames/it10_shipout_PLAN.png")

    h = 940
    tiles = []
    for im, cap in ((sketch, "the sketch: plan at peak stock"),
                    (peak, "built world, peak stock  (warehouse_v2.world.sdf)"),
                    (ship, "built world, after the ship-out  (warehouse_v2_shipout.world.sdf)")):
        w = int(im.width * h / im.height)
        im = im.resize((w, h))
        t = Image.new("RGB", (w, h + 34), "#11171c")
        t.paste(im, (0, 34))
        d = ImageDraw.Draw(t)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
        except Exception:
            f = ImageFont.load_default()
        d.text((10, 8), cap, fill="#e9eff4", font=f)
        tiles.append(t)

    W = sum(t.width for t in tiles) + 24
    out = Image.new("RGB", (W, h + 34), "#11171c")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0)); x += t.width + 12
    o = HERE / "figures/sketch_vs_world.png"
    out.save(o)
    print(f"wrote {o}  ({out.width}x{out.height})")


if __name__ == "__main__":
    main()
