from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/reconfiguration_holdout/e3_availability_routing/independent_heading_fields.py"
)
SPEC = importlib.util.spec_from_file_location(
    "reconfiguration_routing_independent_heading_fields", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
FIELDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIELDS)

RUN_SCRIPT = (
    ROOT / "experiments/reconfiguration_holdout/e3_availability_routing/run_experiment.py"
)
RUN_SPEC = importlib.util.spec_from_file_location(
    "reconfiguration_routing_heading_split_run", RUN_SCRIPT
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
RUN = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(RUN)


def _fake_events(headings: tuple[float, ...]) -> dict[str, dict[str, np.ndarray]]:
    result = {}
    for camera in FIELDS.C.CAMERAS:
        n = len(headings)
        result[camera] = {
            "xy": np.column_stack([np.arange(n, dtype=float), np.zeros(n)]),
            "theta": np.asarray(headings, dtype=float),
            "hit": np.zeros(n, dtype=float),
            "score": np.zeros(n, dtype=float),
            "oracle": np.zeros(n, dtype=float),
        }
    return result


def _option(command: list[str], name: str) -> str:
    return command[command.index(name) + 1]


def test_heading_partition_is_four_diagonals_vs_four_cardinals() -> None:
    FIELDS.validate_heading_partition()
    assert FIELDS.EVALUATION_HEADINGS == pytest.approx(
        (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
    )
    assert FIELDS.TRAINING_HEADINGS == pytest.approx(
        (math.pi / 4.0, 3.0 * math.pi / 4.0,
         5.0 * math.pi / 4.0, 7.0 * math.pi / 4.0)
    )
    assert all(
        FIELDS._angular_distance(train, test) >= FIELDS.C.THETA_TOL
        for train in FIELDS.TRAINING_HEADINGS
        for test in FIELDS.EVALUATION_HEADINGS
    )


def test_event_loaders_pass_disjoint_headings_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[float, ...]]] = []

    def fake_load(environment, *, threshold, thetas):
        assert threshold == pytest.approx(0.25)
        calls.append((environment.key, tuple(thetas)))
        return _fake_events(tuple(thetas))

    monkeypatch.setattr(FIELDS.C, "load_events", fake_load)
    train = FIELDS.load_training_events(0.25)
    nominal_truth = FIELDS.load_evaluation_events("L0", 0.25)
    changed_truth = FIELDS.load_evaluation_events("L1", 0.25)

    assert calls == [
        ("L0", FIELDS.TRAINING_HEADINGS),
        ("L0", FIELDS.EVALUATION_HEADINGS),
        ("L1", FIELDS.EVALUATION_HEADINGS),
    ]
    assert np.array_equal(
        train[FIELDS.C.CAMERAS[0]]["theta"], FIELDS.TRAINING_HEADINGS
    )
    assert np.array_equal(
        nominal_truth[FIELDS.C.CAMERAS[0]]["theta"], FIELDS.EVALUATION_HEADINGS
    )
    assert np.array_equal(
        changed_truth[FIELDS.C.CAMERAS[0]]["theta"], FIELDS.EVALUATION_HEADINGS
    )


def test_event_heading_contract_fails_on_leaked_or_missing_heading() -> None:
    leaked = _fake_events(FIELDS.TRAINING_HEADINGS)
    leaked[FIELDS.C.CAMERAS[0]]["theta"][0] = FIELDS.EVALUATION_HEADINGS[0]
    with pytest.raises(RuntimeError, match="heading contract failed"):
        FIELDS._assert_event_headings(leaked, FIELDS.TRAINING_HEADINGS, "training")


def test_e3_field_caches_are_distinct_and_hybrid_freezes_l0_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = tmp_path / "events_L0_diagonal"
    grid = tmp_path / "grid.npz"
    field_dir = tmp_path / "fields"
    priors = {"L0": tmp_path / "priors/L0", "L1": tmp_path / "priors/L1"}
    monkeypatch.setattr(FIELDS, "EVENTS_DIR", events)
    monkeypatch.setattr(FIELDS, "GRID_PATH", grid)
    monkeypatch.setattr(FIELDS, "FIELDS_DIR", field_dir)

    gp = FIELDS.gp_field_command("gp", "L0", priors)
    hybrid_l0 = FIELDS.gp_field_command("hybrid", "L0", priors)
    hybrid_l1 = FIELDS.gp_field_command("hybrid", "L1", priors)

    assert _option(gp, "--events-dir") == str(events)
    assert "--train-prior-dir" not in gp
    assert _option(hybrid_l0, "--events-dir") == str(events)
    assert _option(hybrid_l1, "--events-dir") == str(events)
    assert _option(hybrid_l0, "--train-prior-dir") == str(priors["L0"])
    assert _option(hybrid_l1, "--train-prior-dir") == str(priors["L0"])
    assert _option(hybrid_l0, "--query-prior-dir") == str(priors["L0"])
    assert _option(hybrid_l1, "--query-prior-dir") == str(priors["L1"])
    assert _option(hybrid_l0, "--out") != _option(hybrid_l1, "--out")
    assert "diagonal" in Path(_option(gp, "--out")).name
    assert "diagonal" in Path(_option(hybrid_l1, "--out")).name


def test_fused_fields_calibrates_learned_arms_from_oof_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    xs, ys = RUN.C.working_grid()
    shape = (len(ys), len(xs))
    arms = {arm for arm, _label, _survey in RUN.ROUTING_ARMS if arm != "shortest"}
    per_environment = {
        environment: {
            arm: {
                camera: np.full(shape, 0.25, dtype=float)
                for camera in RUN.C.CAMERAS
            }
            for arm in arms
        }
        for environment in ("L0", "L1")
    }
    training = {
        camera: {
            "xy": np.asarray([[0.0, 0.0]]),
            "hit": np.asarray([0.0]),
        }
        for camera in RUN.C.CAMERAS
    }
    learned_oof = {
        "gp": {
            camera: {"score": np.asarray([0.11, 0.22]), "hit": np.asarray([0.0, 1.0])}
            for camera in RUN.C.CAMERAS
        },
        "hybrid": {
            camera: {"score": np.asarray([0.77, 0.88]), "hit": np.asarray([1.0, 0.0])}
            for camera in RUN.C.CAMERAS
        },
    }
    monkeypatch.setattr(
        RUN.IHF,
        "build_fields",
        lambda threshold, environments: (
            per_environment, training, learned_oof, {"protocol_id": "test"}
        ),
    )
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    def fake_fit_link(score, hit):
        calls.append((np.asarray(score, dtype=float), np.asarray(hit, dtype=float)))
        return 0.0, 0.0

    monkeypatch.setattr(RUN.C, "fit_link", fake_fit_link)
    RUN.fused_fields(0.25, ("L0", "L1"))

    learned_calls = [(score, hit) for score, hit in calls if len(score) == 2]
    assert len(learned_calls) == 2 * len(RUN.C.CAMERAS)
    assert sum(np.array_equal(score, [0.11, 0.22]) for score, _hit in learned_calls) == 4
    assert sum(np.array_equal(score, [0.77, 0.88]) for score, _hit in learned_calls) == 4
    assert all(len(hit) == 2 for _score, hit in learned_calls)
