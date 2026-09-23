# UnembodiedNavigation

ROS 2 and Gazebo implementation for the commissioned external-camera-network thesis.

Start with:

- `AGENTS.md` for the non-negotiable method and evidence boundary;
- `PLAN.md` for the current execution order;
- `docs/COMMISSIONED_SENSOR_MODEL_CONTRACT.md` for the sensor-model contract;
- `experiments/reference_controlled_commissioning_v1/PROTOCOL.md` for commissioning;
- `pipeline/pipeline_lock.json` for current evidence status.

The current observation starts at the raw YOLO bounding-box bottom centre. Visual-hull
observations are not part of the thesis. Availability is the position-only field `q_i(p)`.
Every correction candidate is evaluated with a covariance fitted to its own residuals.

Historical studies and conclusions have been removed from the working repository. They are
recoverable from Git history and the dated cold archive, but they are not thesis evidence.

## Repository layout

- `src/`: runtime ROS packages.
- `experiments/reference_controlled_commissioning_v1/`: current commissioning and model selection.
- `pipeline/`: current locks and detector/dataset provenance.
- `world/`: locked warehouse geometry.
- `figures/`: thesis-facing figures.
- `pipeline/campaign_runner.py`: campaign runner.
- `tests/`: tests for retained thesis code.

Run tests from the repository root:

```bash
python3 -m pytest -q
```

Before starting Gazebo or a campaign, verify that no other run is active:

```bash
pgrep -af "ros2 launch|ign gazebo|campaign_runner"
```

