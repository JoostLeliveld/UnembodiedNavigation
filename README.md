# UnembodiedNavigation

ROS 2 and Gazebo implementation for the thesis on planning with a commissioned external
camera network: a frozen YOLO11n detector and five wall cameras localise a warehouse robot,
and the planner uses each camera's matched runtime covariance to choose routes.

Start with:

- `AGENTS.md` — the method contract, evidence boundary and working rules;
- `docs/METHOD.md` — the authoritative method and its latest amendment;
- `docs/STATE.md` — where the data and results are, current state and open items.

## Layout

- `pipeline/` — the one pipeline: `dataset.py` (the only data loader), `refit.sh` (detector
  inference, gate, correction, R0/R1/R2, runtime package, planning precision), final audit,
  `routes.sh`, `campaign.sh`, `score_collisions.py`, `analyze_campaign.py`, the campaign
  templates and `tasks.yaml`; `capture/`, `detector/`, `decisions/` and `ops/` hold the
  capture, detector-training, decision and operations provenance.
- `world/` — the warehouse description; the world is
  `src/sim/gazebo_worlds/worlds/warehouse_v2.world.sdf`.
- `figures/` — one generator per manuscript figure.
- `src/` — ROS runtime packages; `config/sensor_gate.yaml` — the sensor gate.
- `tests/` — tests for everything above.
- `logs/` (not in git) — data and results; see `docs/STATE.md`.

## Running

```bash
python3 -m pytest -q          # with /opt/ros/humble and install/ sourced
bash pipeline/refit.sh        # reference fits into logs/thesis/fits
```

Before starting Gazebo or a campaign, verify that no other run is active:

```bash
pgrep -af "ros2 launch|ign gazebo|campaign_runner"
```
