#!/usr/bin/env python3
"""Verify and summarize the separately frozen one-arm controller follow-up."""
import json
from pathlib import Path

import network_navigation_analysis as nav


def main():
    out = nav.OUT / 'network_navigation_recovery_evidence'
    selected = json.loads((out / 'selection.json').read_text())
    results = json.loads((out / 'results.json').read_text())
    protocol = json.loads((out / 'protocol.json').read_text())
    if results['selection_sha256'] != nav.digest(out / 'selection.json'):
        raise ValueError('Results have a different selection')
    if len(selected['runs']) != 1 or selected['runs'][0]['arm'] != 'P0':
        raise ValueError('Expected the single registered P0 follow-up')
    entry = selected['runs'][0]
    run = nav.REPO / entry['run']
    for name, expected in entry['files'].items():
        if nav.digest(run / name) != expected:
            raise ValueError(f'Frozen input changed: {name}')
    # These are frozen component snapshots, not a claim to hash every dependency.
    checked_sources = {}
    for name, expected in protocol['sources'].items():
        archived = out / 'source_snapshot' / name
        source = archived if archived.exists() else nav.REPO / name
        if nav.digest(source) != expected:
            raise ValueError(f'Protocol source changed: {name}')
        checked_sources[name] = dict(sha256=expected, snapshot=archived.exists())
    events = nav.aligned.assimilations(run)
    observations = nav.aligned.observations(run)
    if {r['source_batch_id'] for r in events} != {r['source_batch_id'] for r in observations}:
        raise ValueError('Recorded fused events do not reconcile')
    console = out / 'execution_console.log'
    activation_lines = [line for line in console.read_text().splitlines()
                        if '[hierarchical] checked rotation recovery:' in line]
    r = results['results'][0]
    summary = json.loads((run / 'run_summary.json').read_text())
    audit = dict(
        selection_sha256=nav.digest(out / 'selection.json'),
        results_sha256=nav.digest(out / 'results.json'),
        protocol_sha256=nav.digest(out / 'protocol.json'),
        console_sha256=nav.digest(console),
        analysis_source_sha256=nav.digest(Path(__file__)),
        registered_run=entry['run'], protocol_components=checked_sources,
        recorded_camera_rows=len(observations),
        recorded_fused_events=len({r['source_batch_id'] for r in observations}),
        terminal_outcomes=len(events),
        rotation_recovery_log_count=len(activation_lines), activation_lines=activation_lines,
        outcome=r['status'], collision_contact=summary['collision_contact'],
        collision_geom=summary['collision_geom'],
        contact_topic_publishers=summary['contact_topic_publishers'],
        contact_messages_seen=summary['contact_messages_seen'],
        scope='One development drive. Recorded-event reconciliation is not proof of complete upstream opportunity accounting or crash-atomic updates.',
    )
    (out / 'recovery_verification.json').write_text(json.dumps(audit, indent=2) + '\n')
    text = f'''# Guarded-controller follow-up

The single registered P0/seed-210 drive ended with **{r['status']}** and
**{len(activation_lines)} logged checked-rotation activation(s)**. This exercises the opt-in
recovery branch through the unchanged clearance check. No contact or geometry collision
was recorded. The logger saw {summary['contact_topic_publishers']} contact publishers and
{summary['contact_messages_seen']} contact messages; this is not an independent contact-sensor
liveness validation.

| Diagnostic | Measured value |
|---|---:|
| Unique belief timestamps | {r['belief_samples']} |
| Belief XY median / p95 error | {r['belief_position_median_cm']:.2f} / {r['belief_position_p95_cm']:.2f} cm |
| Belief heading p95 error | {r['belief_heading_p95_deg']:.2f} degrees |
| Nominal planar 95%-ellipse coverage | {100*r['planar_95_ellipse_coverage']:.2f}% |
| Longest accepted-correction gap | {r['longest_correction_gap_s']:.3f} s |
| Dropped fused corrections | {r['assimilation_status_counts']['dropped']} / {r['assimilations']} ({100*r['correction_dropped_fraction']:.2f}%) |
| Simulated driving time / reference path length | {r['duration_sim_s']:.3f} s / {r['path_length_m']:.2f} m |
| Camera means and full R checked exactly | {r['camera_model_audit']['observations_checked']} |

Errors use each unique published belief timestamp, from first command to stop, against
interpolated `gt_stamp`. Heading is wrapped and planar coverage uses the two-dimensional
chi-square threshold. The simulator reference is timestamped at logger receipt; the
alignment does not eliminate unknown reference transport delay. These time samples are
correlated and are not independent repetitions. Coverage of 100% is not calibration proof.

The same uniform planner field, frozen NN minus per-camera offset, full constant camera R,
Q, robust fusion and mission were retained. Controller recovery and additional command
fatal-stop/negative-age guards are declared separately from the corrected three-arm pilot.
The source packet passed 159 focused tests with three absent historical traces skipped.
Different optimized routes, callback timings and per-message noise draws prevent a paired
causal comparison with the earlier P0. No learned-field improvement is inferred.

`recovery_verification.json` verifies raw selection hashes, the protocol's listed components,
recorded fused-event reconciliation and the retained console activation. It does not cover
unlisted dependencies or establish complete pre-fusion admission logging. The source-pinned
module audits identify additional runtime defects that this scoped check does not exclude.

Reproduce analysis from the robot repository:

```bash
MPLCONFIGDIR=/tmp/icra_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/network_navigation_analysis.py navigation --config experiments/icra_commissioning/network_navigation_recovery_pilot.yaml --campaign logs/studies/icra_commissioning_20260905/network_navigation_recovery_pilot --out logs/studies/icra_commissioning_20260905/network_navigation_recovery_evidence
MPLCONFIGDIR=/tmp/icra_mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 experiments/icra_commissioning/network_recovery_report.py
```

[Frozen selection](selection.json), [measured results](results.json),
[verification](recovery_verification.json), [vector figure](navigation_pilot.pdf).
'''
    (out / 'recovery_result.md').write_text(text)
    print(json.dumps({k: audit[k] for k in ('outcome', 'rotation_recovery_log_count',
                    'recorded_fused_events', 'terminal_outcomes')}))


if __name__ == '__main__':
    main()
