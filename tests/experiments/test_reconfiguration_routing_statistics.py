from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "experiments/reconfiguration_holdout/e3_availability_routing/summarize_cells.py"
)
SPEC = importlib.util.spec_from_file_location("reconfiguration_routing_statistics", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
STATS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(STATS)


def test_exact_two_sided_sign_test_drops_ties() -> None:
    positive, negative, ties, p_value = STATS.exact_two_sided_sign_test(
        [2.0, 1.0, 0.25, -0.5, 0.0, 5e-13]
    )
    assert (positive, negative, ties) == (3, 1, 2)
    assert p_value == pytest.approx(0.625)


def test_holm_adjust_matches_step_down_reference_and_original_order() -> None:
    # Sorted p-values are .01, .03, .04, so Holm gives .03, .06, .06.
    adjusted = STATS.holm_adjust([0.04, 0.01, 0.03])
    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def _write_synthetic_design(
    tmp_path: Path,
    *,
    drop_last: bool = False,
    changed_environments: tuple[str, ...] = ("L1",),
) -> tuple[Path, Path]:
    subsets = ("4", "3", "2_opposite", "2_same_wall", "1")
    budgets = (0.05, 0.10, 0.20, 0.35, 0.50)
    tasks = tuple(f"task_{index}" for index in range(6))
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "subsets": subsets,
                "budgets": budgets,
                "tasks": tasks,
                "arms": ["shortest", "gp", "mono_depth"],
                "environments": ["L0", *changed_environments],
            }
        ),
        encoding="utf-8",
    )

    rows = []
    for subset in subsets:
        for budget in budgets:
            for task_index, task in enumerate(tasks):
                # GP starts 1 m better in L0 but degrades 1.5 m more.  Therefore
                # DiD = +1.5 m and the direct L1 gap is +0.5 m for every task.
                environment_values = [("L0", 9.0 + task_index, 10.0 + task_index)]
                for changed_index, environment in enumerate(changed_environments, start=1):
                    environment_values.append(
                        (
                            environment,
                            10.5 + changed_index + task_index,
                            10.0 + changed_index + task_index,
                        )
                    )
                for environment, gp, mono in environment_values:
                    rows.extend(
                        [
                            {
                                "environment": environment,
                                "subset": subset,
                                "task": task,
                                "budget": budget,
                                "arm": "gp",
                                "blind_true_m": gp,
                            },
                            {
                                "environment": environment,
                                "subset": subset,
                                "task": task,
                                "budget": budget,
                                "arm": "mono_depth",
                                "blind_true_m": mono,
                            },
                        ]
                    )
    if drop_last:
        rows.pop()

    routes_path = tmp_path / "e3_routes.csv"
    with routes_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "environment",
                "subset",
                "task",
                "budget",
                "arm",
                "blind_true_m",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    return routes_path, manifest_path


def test_summary_uses_all_25_cells_and_pairs_the_four_observations(tmp_path: Path) -> None:
    routes_path, manifest_path = _write_synthetic_design(tmp_path)
    rows = STATS.summarize_routes(
        routes_path, manifest_path, bootstrap_resamples=200, bootstrap_seed=7
    )

    assert len(rows) == 25
    first = rows[0]
    assert first["changed_environment"] == "L1"
    assert (first["subset"], first["budget"], first["n_tasks"]) == ("4", 0.05, 6)
    assert first["mean_gp_degradation_m"] == pytest.approx(2.5)
    assert first["mean_mono_depth_degradation_m"] == pytest.approx(1.0)
    assert first["mean_did_m"] == pytest.approx(1.5)
    assert (first["did_positive"], first["did_negative"], first["did_ties"]) == (6, 0, 0)
    assert first["did_sign_p_raw"] == pytest.approx(0.03125)
    # Twenty-five equal p-values: each Holm-adjusted value is 25 * .03125.
    assert first["did_sign_p_holm_25"] == pytest.approx(0.78125)
    assert first["did_holm_reject_0_05"] is False
    assert first["mean_changed_gp_minus_mono_depth_m"] == pytest.approx(0.5)


def test_summary_fails_closed_when_one_member_of_a_pair_is_missing(tmp_path: Path) -> None:
    routes_path, manifest_path = _write_synthetic_design(tmp_path, drop_last=True)
    with pytest.raises(ValueError, match="missing 1 paired observations"):
        STATS.summarize_routes(
            routes_path, manifest_path, bootstrap_resamples=20, bootstrap_seed=3
        )


def test_write_summary_persists_contract_and_input_hashes(tmp_path: Path) -> None:
    routes_path, route_manifest_path = _write_synthetic_design(tmp_path)
    out_path = tmp_path / "e3_cell_summary.csv"
    out_manifest_path = tmp_path / "e3_cell_summary_manifest.json"

    STATS.write_summary(
        routes_path,
        route_manifest_path,
        out_path,
        out_manifest_path,
        bootstrap_resamples=50,
        bootstrap_seed=11,
    )

    with out_path.open(newline="", encoding="utf-8") as handle:
        persisted = list(csv.DictReader(handle))
    metadata = json.loads(out_manifest_path.read_text(encoding="utf-8"))
    assert len(persisted) == 25
    assert tuple(persisted[0]) == STATS.SUMMARY_COLUMNS
    assert metadata["n_cells"] == 25
    assert metadata["experimental_unit"] == "start_goal_task"
    assert metadata["changed_environment"] == "L1"
    assert "single fixed L0-to-L1 layout" in metadata["inferential_scope"]
    assert metadata["primary_multiplicity"].startswith("Holm")
    assert metadata["input_routes_sha256"] == STATS.sha256(routes_path)
    assert metadata["output_csv_sha256"] == STATS.sha256(out_path)


def test_one_route_table_can_emit_environment_keyed_summaries(tmp_path: Path) -> None:
    routes_path, manifest_path = _write_synthetic_design(
        tmp_path, changed_environments=("L1", "L2")
    )
    l1 = STATS.summarize_routes(
        routes_path, manifest_path, changed_environment="L1",
        bootstrap_resamples=40, bootstrap_seed=5,
    )
    l2 = STATS.summarize_routes(
        routes_path, manifest_path, changed_environment="L2",
        bootstrap_resamples=40, bootstrap_seed=5,
    )
    assert len(l1) == len(l2) == 25
    assert {row["changed_environment"] for row in l1} == {"L1"}
    assert {row["changed_environment"] for row in l2} == {"L2"}
    assert l1[0]["mean_gp_degradation_m"] == pytest.approx(2.5)
    assert l2[0]["mean_gp_degradation_m"] == pytest.approx(3.5)
