from typing import Any, Dict
import os

import yaml


def load_experiment_presets(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise RuntimeError(f"experiments.yaml not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    presets = data.get("experiments")
    if not isinstance(presets, dict) or not presets:
        raise RuntimeError("experiments.yaml must contain a non-empty 'experiments' mapping")
    return presets


def select_experiment(presets: Dict[str, Any], experiment_name: str) -> Dict[str, Any]:
    if not experiment_name:
        known = ", ".join(sorted(presets.keys()))
        raise RuntimeError(
            f"'experiment' launch arg is required. Available presets: {known or 'none'}"
        )
    if experiment_name not in presets:
        known = ", ".join(sorted(presets.keys()))
        raise RuntimeError(
            f"Unknown experiment preset '{experiment_name}'. Available: {known or 'none'}"
        )

    preset = presets[experiment_name]
    if not isinstance(preset, dict):
        raise RuntimeError(f"Preset '{experiment_name}' must be a mapping")

    launch_name = str(preset.get("launch", "")).strip()
    if launch_name not in {"boundary_only", "boundary_only_agent"}:
        raise RuntimeError(
            f"Preset '{experiment_name}' must define launch as "
            "'boundary_only' or 'boundary_only_agent'"
        )

    args = preset.get("args", {})
    if not isinstance(args, dict):
        raise RuntimeError(f"Preset '{experiment_name}' field 'args' must be a mapping")

    return {"launch": launch_name, "args": args}

