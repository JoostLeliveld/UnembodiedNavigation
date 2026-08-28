"""Shared loading for the uncertainty figures: sightings with their geometry attached."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sources as src            # noqa: E402
import style as D                # noqa: E402
sys.path.insert(0, str(D.REPO / "experiments/measurement_commissioning"))
from camera import camera_models  # noqa: E402
from observation import jacobian  # noqa: E402

OUT = D.REPO / "logs/studies/deck_figures/uncertainty"
OUT.mkdir(parents=True, exist_ok=True)
DATA = D.REPO / "logs/perception_datasets/warehouse_v2_yolo_shared_20260822"


def rows():
    cal = src.calibration()["calibration"]
    cu, cv = cal["coefficients_du"], cal["coefficients_dv"]
    cams = camera_models(DATA)
    out = []
    for r in src.sightings():
        x, y, yaw, rng = (float(r["x"]), float(r["y"]), float(r["yaw"]), float(r["range_m"]))
        J = jacobian(cams[r["camera"]], x, y, yaw)
        if abs(np.linalg.det(J)) < 1e-9:
            continue
        px = np.array([float(r["du_px"]) - np.polyval(cu[::-1], rng),
                       float(r["dv_px"]) - np.polyval(cv[::-1], rng)])
        Ji = np.linalg.inv(J)
        out.append({"camera": r["camera"], "x": x, "y": y, "yaw": yaw, "range_m": rng,
                    "px": px, "Jinv": Ji, "ground_cm": (Ji @ px) * 100.0})
    return out, cal, cams


def ladder():
    return json.loads((D.REPO / "logs/studies/measurement_commissioning/uncertainty_ladder.json").read_text())


BANDS = ((0, 6), (6, 9), (9, 12), (12, 15), (15, 18), (18, 25))
