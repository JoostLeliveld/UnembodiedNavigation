# REUSE_MAP — geometric_baseline (C0)

C0 adds **no new runtime module**. It is a new `global_planner_mode` branch plus
condition plumbing; every load-bearing component is reused unchanged.

## Reused unchanged

| Asset | Path | Role for C0 |
|---|---|---|
| `generate_route_seeds` | `src/unav_common/unav_common/lane_graph_routes.py` | Enumerates the condition-neutral lane-graph route candidates over the driveable region. C0 picks the **shortest** by total polyline length. Same call, same start=`m0[:2]`, same goal as the C1/C2 EFE seed generation. |
| Geometric local tracker (`turn_then_go` via `_dispatch_local_controller`) | `src/planning/planning/nodes/{efe_agent_node,unicycle_planner_node}.py` | Follows the chosen route. Identical to C1/C2's LOCAL phase — unchanged. |
| `NogoZoneCostModel` | `src/planning/planning/core/nogo_cost.py` (used via `planners/base_planner.py`) | Shared feasibility / no-go geometry layer. Same driveable + no-go region as C1/C2. |
| Campaign harness | `scripts/visibility_comparison/run_visibility_campaign.py` | 4 routes × 5 seeds matrix, launch orchestration, straggler cleanup, outcome audit — reused as-is (honest_campaign machinery). |
| Locked perception + noise + GT stack | detector `warehouse_yolo_detector_v1`, command/encoder noise, GT bridge | Identical to v1; C0 still localises from the camera (it just ignores reliability for planning). |

## New / changed (minimal)

| Change | File | What |
|---|---|---|
| `global_planner_mode` param + GLOBAL geometric branch | `src/planning/planning/nodes/efe_agent_node.py` | New param (default `efe`); when `geometric_shortest_path`, pick shortest route seed and hand to local tracker, **skipping the EFE solve**. |
| Condition registration | `scripts/visibility_comparison/run_visibility_campaign.py` | `C0 -> geometric_shortest_path`; GP artifact gated to `visibility_aware_efe` only; `global_planner_mode` added to forwarded launch args. |
| Planner allow-list + mapping | `src/experiments/launch/warehouse_primary_comparison.launch.py` | `geometric_shortest_path` allowed; sets `use_visibility_model=False`, `global_planner_mode`, EFE terms off. |
| Plumbing + guards + a logger fix | `src/experiments/experiments/core/visibility_launch_common.py` | Plumb `global_planner_mode`; GP-artifact requirement restricted to `visibility_aware_efe`; agent allow-list extended. Also fixed: `terminate_on_geom_collision` was set in cfg but never forwarded to the experiment_logger node — now forwarded (needed by v2's `terminate_on_geom_collision: false`). |
| New campaign config | `scripts/visibility_comparison/warehouse_visibility_campaign_honest_v2.yaml` | v1 verbatim + C0 condition + `terminate_on_geom_collision: false` + `ros_domain_id_base` lowered to fit 60 runs. |

## Not reused (deliberately)

- The GP reliability artifact (`gp_artifact`) — C0 is camera-model-free and must
  not receive it (enforced in both the campaign runner and the launch guards).
- The one-shot global EFE solve (`global_planner.plan`) — skipped entirely for C0.
