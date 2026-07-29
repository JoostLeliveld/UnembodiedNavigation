"""Method-critical campaign settings must not silently diverge between worlds.

`run_visibility_campaign.py` loads each YAML with a bare `yaml.safe_load` -- no
`extends`, no includes -- so all 30+ configs restate the full method config as
flat copies. That is exactly how `nogo_weight` drifted 2000 -> 1200 in the 4-cam
world while every config "looked identical", and the drift went unnoticed until
a parity audit (docs/multicam_vs_paper1_correction_parity.md).

Until the configs get a base+overlay, this test is the guard: the settings below
must agree across worlds, and anything allowed to differ must be listed here
with a reason. It fails on NEW drift, which is the failure mode that hurt.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


CONFIG_DIR = Path(__file__).resolve().parents[2] / "scripts" / "visibility_comparison"

#: Settings that encode the METHOD and must be identical in every world.
#: A world may legitimately differ in geometry, tasks, cameras or detector --
#: not in how the belief filter gates a correction, nor in how hard the planner
#: is pushed away from obstacles.
METHOD_CRITICAL_KEYS = (
    "pixel_max_correction_jump_m",
    "pixel_correction_nis_threshold",
    "pixel_timeout_s",
    "skip_stale_pixel_correction",
    "nogo_weight",
    "v_max",
    "r_visible_uv",
)

#: Keys that legitimately differ, each with the reason. Anything not listed here
#: and not identical across worlds fails the test.
ALLOWED_TO_DIFFER = {
    # Open-floor 4-cam world uses obstacle footprints (keep_out); paper-1's
    # marked-lane world uses keep_in. Verified correct, see the parity audit's
    # D1 correction.
    "nogo_mode",
}


def load_configs():
    out = []
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        try:
            cfg = yaml.safe_load(path.read_text())
        except yaml.YAMLError as exc:                      # pragma: no cover
            pytest.fail(f"{path.name}: {exc}")
        if isinstance(cfg, dict) and cfg.get("world"):
            out.append((path.name, cfg))
    return out


CONFIGS = load_configs()


def world_family(cfg) -> str:
    return "4cam" if str(cfg["world"]).startswith("warehouse_full_4cam") else "paper1"


def test_configs_were_found():
    assert len(CONFIGS) > 20
    families = {world_family(cfg) for _, cfg in CONFIGS}
    assert families == {"4cam", "paper1"}, families


@pytest.mark.parametrize("key", METHOD_CRITICAL_KEYS)
def test_method_critical_setting_agrees_across_worlds(key):
    by_value: dict[object, list[str]] = {}
    for name, cfg in CONFIGS:
        if key in cfg:
            by_value.setdefault(_hashable(cfg[key]), []).append(name)
    if not by_value:
        pytest.skip(f"{key} not set in any campaign config")
    assert len(by_value) == 1, (
        f"{key} diverged across campaign configs: "
        + "; ".join(
            f"{value!r} in {len(names)} config(s) e.g. {sorted(names)[:3]}"
            for value, names in by_value.items()
        )
        + ". Flat YAML copies drift -- fix the outlier, or move the key to "
          "ALLOWED_TO_DIFFER with a reason."
    )


def test_allowed_differences_are_documented_and_real():
    """Keep the allowlist honest: no stale entries that actually agree."""
    for key in ALLOWED_TO_DIFFER:
        values = {_hashable(cfg[key]) for _, cfg in CONFIGS if key in cfg}
        assert len(values) > 1, (
            f"{key} is on the ALLOWED_TO_DIFFER list but no longer differs "
            f"({values}); remove it so the guard stays tight."
        )


def test_nogo_weight_matches_paper1_everywhere():
    """The specific drift the parity audit found; pinned so it cannot return."""
    for name, cfg in CONFIGS:
        if "nogo_weight" in cfg:
            assert float(cfg["nogo_weight"]) == 2000.0, name


def test_4cam_configs_enable_the_prediction_plausibility_cap():
    for name, cfg in CONFIGS:
        if world_family(cfg) != "4cam":
            continue
        cap = cfg.get("max_predict_speed_mps")
        assert cap is not None, f"{name}: 4-cam config must set max_predict_speed_mps"
        assert float(cap) == pytest.approx(float(cfg["v_max"])), name


def test_paper1_configs_leave_the_prediction_cap_off():
    """honest_campaign_v1 is locked; it must stay reproducible."""
    for name, cfg in CONFIGS:
        if world_family(cfg) != "paper1":
            continue
        assert float(cfg.get("max_predict_speed_mps", 0.0)) == 0.0, name


def test_multicam_2x2_rerun_freezes_the_paper1_covariance_profile():
    rerun_configs = {
        "_mc_2x2_fus_gp.yaml",
        "_mc_2x2_fus_nogp.yaml",
        "_mc_2x2_sel_gp.yaml",
        "_mc_2x2_sel_nogp.yaml",
    }
    found = {name: cfg for name, cfg in CONFIGS if name in rerun_configs}
    assert set(found) == rerun_configs
    for name, cfg in found.items():
        assert cfg["manager_covariance_profile"] == "paper1_historical", name
        assert float(cfg["r_visible_uv"]) == pytest.approx(2.5), name
        assert float(cfg["r_miss_uv"]) == pytest.approx(40.0), name


def _hashable(value):
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, dict):
        return tuple(sorted(value.items()))
    return value


# --------------------------------------------------------------------------
# per_camera mode: the manager MUST publish, or the run silently dead-reckons
# --------------------------------------------------------------------------

LAUNCH_COMMON = (
    Path(__file__).resolve().parents[2]
    / "src" / "experiments" / "experiments" / "core" / "visibility_launch_common.py"
)


def test_per_camera_configs_do_not_disable_the_manager_publisher():
    """In per_camera mode the planner's ONLY correction source is that topic.

    If camera_manager does not publish it, the planner gets zero corrections and
    dead-reckons the whole run -- with no error, because an absent topic looks
    exactly like a camera that never detects.
    """
    for name, cfg in CONFIGS:
        if str(cfg.get("state_correction_mode", "fused")) != "per_camera":
            continue
        explicit = cfg.get("manager_publish_map_observations")
        assert explicit is not False, (
            f"{name}: state_correction_mode=per_camera but "
            f"manager_publish_map_observations is false -- the planner would "
            f"receive no corrections at all"
        )


def test_launch_derives_the_publisher_from_the_correction_mode():
    """Guard the default that makes the invariant above hold without config work."""
    src = LAUNCH_COMMON.read_text()
    assert "'publish_map_observations'" in src
    derivation = src.split("'publish_map_observations'", 1)[1][:300]
    assert "manager_publish_map_observations" in derivation
    assert "state_correction_mode" in derivation
    assert "per_camera" in derivation


def test_state_correction_mode_values_are_valid():
    for name, cfg in CONFIGS:
        mode = str(cfg.get("state_correction_mode", "fused"))
        assert mode in ("fused", "per_camera"), f"{name}: {mode!r}"


def test_reproduction_arms_stay_fused_and_dev_configs_are_per_camera():
    """The 2x2 historical reproduction is a fused-path experiment by definition."""
    for name, cfg in CONFIGS:
        mode = str(cfg.get("state_correction_mode", "fused"))
        if name.startswith("_mc_2x2_"):
            assert mode == "fused", f"{name}: reproduction arm must stay fused"
        if name.startswith("_rob_"):
            assert mode == "per_camera", f"{name}: dev robustness config should be per_camera"
