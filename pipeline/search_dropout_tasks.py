"""Usage: python3 pipeline/search_dropout_tasks.py ROUTE_PLANNING_CONFIG CAMERA_ID

Screen start/goal pairs on the lane grid for a camera-dropout task where information decides.

Wanted, for the dropped camera REMOVED:
  - start and goal pass the task-visibility rule without REMOVED;
  - every model, all cameras, prefers route P (the cheapest by the exact planner objective);
  - global and per-camera keep P after the dropout;
  - the spatial model, after the dropout, prefers a different route Q.
Routes are lane-grid paths; each is scored as a fixed rollout with the planner's own objective
(evaluate_rollout_controls), not optimized. The best candidates are then checked with plan().
"""
import itertools, json, math, sys
from pathlib import Path
import numpy as np, yaml, networkx as nx

REPO = Path('/home/joostleliveld/Thesis/UnembodiedNavigation_v9')
sys.path[:0] = [str(REPO / 'pipeline'), str(REPO), str(REPO / 'src/unav_common')]
import solve_routes as S
from pipeline.score_collisions import DriveableRegion
from pipeline.check_task_visibility import Views, THRESHOLD

REMOVED = sys.argv[2] if len(sys.argv) > 2 else 'camera_B'
campaign = yaml.safe_load(open(sys.argv[1]))
region, views = DriveableRegion.for_world(), Views()
XS = [-10.875, -6.95, -3.05, 0.975, 10.775]
YS = [-8.725, -6.25, -5.75, -1.175, 2.425, 6.35, 8.625]


def seg_clear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.hypot(*(b - a)); yaw = math.atan2(b[1] - a[1], b[0] - a[0]); n = max(int(L / 0.05), 1)
    return all(min(region.clearance(tuple(a + (b - a) * k / n) + (yaw,))) >= 0.0 for k in range(n + 1))


G = nx.Graph()
nodes = [(x, y) for x in XS for y in YS if seg_clear((x, y), (x + 1e-3, y)) and seg_clear((x, y), (x, y + 1e-3))]
G.add_nodes_from(nodes)
for x in XS:
    col = sorted(n for n in nodes if n[0] == x)
    for a, b in zip(col, col[1:]):
        if seg_clear(a, b):
            G.add_edge(a, b, w=abs(b[1] - a[1]))
for y in YS:
    row = sorted(n for n in nodes if n[1] == y)
    for a, b in zip(row, row[1:]):
        if seg_clear(a, b):
            G.add_edge(a, b, w=abs(b[0] - a[0]))
visible = {n: views.best_active(n[0], n[1], REMOVED)[1] >= THRESHOLD for n in nodes}
print(f'{len(nodes)} nodes, {G.number_of_edges()} edges, {sum(visible.values())} visible without {REMOVED}', flush=True)

collision = S.serialize_collision_geometry_from_world(
    str(REPO / 'src/sim/gazebo_worlds/worlds' / campaign['world']),
    *[tuple(p) for p in (yaml.safe_load(open(REPO / campaign['world_profiles']))['worlds']['warehouse_v2.world.sdf'][k]
                         for k in ('collision_model_names', 'collision_include_names'))],
    yaml.safe_load(open(REPO / campaign['world_profiles']))['worlds']['warehouse_v2.world.sdf'])
boundary = S.serialize_driveable_geometry_from_profile(
    yaml.safe_load(open(REPO / campaign['world_profiles']))['worlds']['warehouse_v2.world.sdf'])
planners = {}
for model in ('global', 'per_camera', 'spatial'):
    for state in ('intact', 'removal'):
        cfg = campaign['conditions'][f'{model}_{state}']
        active = tuple(c for c in S.CAMERAS if not (state == 'removal' and c == REMOVED))
        planners[(model, state)] = S.planner(Path(cfg['camera_network_artifact_path']), collision, boundary, active, campaign)
prior = np.diag([0.05 ** 2, 0.05 ** 2, math.radians(5.0) ** 2])


def score(start, goal, path):
    m0 = np.array([start[0], start[1], math.atan2(path[1][1] - start[1], path[1][0] - start[0])])
    out = {}
    for key, pl in planners.items():
        u = pl._controls_for_waypoints(m0, [list(p) for p in path[1:]])
        r = pl.evaluate_rollout_controls(m0, prior, np.asarray(goal, float), u)
        if not r['rollout_valid'] or r['terminal_goal_distance_pred'] > 0.3:
            return None
        out[key] = r['total_cost']
    return out


sys.path.insert(0, str(REPO / 'figures'))
import paper as P
from scipy.interpolate import RegularGridInterpolator
_xs, _ys, _info = P.network_information('spatial', REMOVED)
_field = RegularGridInterpolator((np.asarray(_ys), np.asarray(_xs)), np.asarray(_info), bounds_error=False, fill_value=0.0)


def blind_m(path):
    pts = []
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a, float), np.asarray(b, float); n = max(int(np.hypot(*(b - a)) / 0.1), 1)
        pts += [a + (b - a) * k / n for k in range(n)]
    v = _field(np.asarray(pts)[:, ::-1]) < 10.0
    run = best = 0
    for x in v:
        run = run + 1 if x else 0; best = max(best, run)
    return 0.1 * best


results = []
pairs = [(s, g) for s, g in itertools.permutations(nodes, 2)
         if visible[s] and visible[g] and nx.has_path(G, s, g) and abs(s[0] - g[0]) + abs(s[1] - g[1]) >= 8]
print(len(pairs), 'visible pairs', flush=True)
for s, g in pairs:
    try:
        paths = list(itertools.islice(nx.shortest_simple_paths(G, s, g, weight='w'), 6))
    except nx.NetworkXNoPath:
        continue
    L0 = nx.path_weight(G, paths[0], 'w')
    paths = [p for p in paths if nx.path_weight(G, p, 'w') <= 1.4 * L0]
    if len(paths) < 2 or blind_m(paths[0]) < 3.0 or not any(blind_m(p) == 0.0 for p in paths[1:]):
        continue
    paths = [paths[0]] + [p for p in paths[1:] if blind_m(p) == 0.0][:2]
    scored = [(p, score(s, g, p)) for p in paths]
    scored = [(p, c) for p, c in scored if c]
    if len(scored) < 2:
        continue
    pick = {k: min(range(len(scored)), key=lambda i: scored[i][1][k]) for k in planners}
    base = pick[('global', 'intact')]
    same = all(pick[k] == base for k in planners if k != ('spatial', 'removal'))
    if not same or pick[('spatial', 'removal')] == base:
        continue
    q = pick[('spatial', 'removal')]
    # margins: how firmly the intact choice holds, and how firmly spatial switches
    margin_keep = min(min(c[k] for j, (_, c) in enumerate(scored) if j != base) - scored[base][1][k]
                      for k in planners if k != ('spatial', 'removal'))
    margin_switch = scored[base][1][('spatial', 'removal')] - scored[q][1][('spatial', 'removal')]
    results.append({'start': s, 'goal': g, 'P': scored[base][0], 'Q': scored[q][0],
                    'len_P': nx.path_weight(G, scored[base][0], 'w'), 'len_Q': nx.path_weight(G, scored[q][0], 'w'),
                    'margin_keep': margin_keep, 'margin_switch': margin_switch})
    print('HIT', s, g, 'lenP %.1f lenQ %.1f keep %.3f switch %.3f' % (results[-1]['len_P'], results[-1]['len_Q'], margin_keep, margin_switch), flush=True)
results.sort(key=lambda r: -min(r['margin_keep'], r['margin_switch']))
json.dump(results, open(Path(__file__).with_name(f'task_search_{REMOVED}.json'), 'w'), indent=1, default=list)
print('done', len(results), 'hits')
