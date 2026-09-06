#!/usr/bin/env python3
"""Build a descriptive repair ledger from three explicitly frozen pilot selections."""
import json
from pathlib import Path

import network_navigation_analysis as nav


STAGES = (
    ('Original tracking settings', 'network_navigation_evidence'),
    ('Denser waypoints', 'network_navigation_tracking_evidence'),
    ('Corrected runtime', 'network_navigation_runtime_evidence'),
)


def main():
    out = nav.OUT/'network_navigation_runtime_evidence'
    sources = {}; groups = []
    for label, name in STAGES:
        root=nav.OUT/name
        selection_path=root/'selection.json'; results_path=root/'results.json'
        selection=json.loads(selection_path.read_text()); results=json.loads(results_path.read_text())
        if results['selection_sha256'] != nav.digest(selection_path):
            raise ValueError('Results do not match frozen selection')
        if len(selection['runs']) != 3 or {r['arm'] for r in selection['runs']} != set(nav.ARMS):
            raise ValueError('Incomplete pilot')
        for entry in selection['runs']:
            for filename, expected in entry['files'].items():
                if nav.digest(nav.REPO/entry['run']/filename) != expected:
                    raise ValueError('Selected raw input changed')
            # This enforces the repository event-accounting contract again.
            nav.aligned.assimilations(nav.REPO/entry['run'])
        groups.append((label,results['results']))
        for p in (selection_path,results_path,root/'protocol.json'):
            sources[str(p.relative_to(nav.REPO))]=nav.digest(p)

    rows = [
        '# Runtime integration evidence', '',
        'Three separately frozen development pilots, one seed (210) per field in each. '
        'Every attempt is retained. Stage changes are tracking/software corrections, '
        'not learned-field effects. Different drives have different durations and images. '
        'Equal seeds do not guarantee identical disturbances because actuation noise advances per command message.', '',
        'Belief errors use each unique belief timestamp and interpolated `gt_stamp`, '
        'from first command to stop. Position is Euclidean XY error in centimetres; '
        'heading error is wrapped. Planar coverage uses chi-square dimension 2; full-pose '
        'coverage uses dimension 3. Time samples are not independent replications.', '',
        '| Stage | Field | Outcome | Unique beliefs | XY median / p95 [cm] | Heading p95 / final [deg] | XY / pose 95% coverage | Longest correction gap [s] | Dropped [%] | Time [sim s] | Path [m] |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for label, values in groups:
        for r in values:
            if r['status']=='infrastructure_invalid':
                rows.append(f"| {label} | {nav.NAMES[r['arm']]} | Infrastructure invalid: {r['reason']} | — | — | — | — | — | — | — | — |")
                continue
            rows.append(f"| {label} | {nav.NAMES[r['arm']]} | {r['status']} | {r['belief_samples']} | "
                f"{r['belief_position_median_cm']:.2f} / {r['belief_position_p95_cm']:.2f} | "
                f"{r['belief_heading_p95_deg']:.2f} / {r['belief_heading_final_deg']:.2f} | "
                f"{100*r['planar_95_ellipse_coverage']:.1f}% / {100*r['pose_95_ellipsoid_coverage']:.1f}% | "
                f"{r['longest_correction_gap_s']:.2f} | {100*r['correction_dropped_fraction']:.2f} | "
                f"{r['duration_sim_s']:.2f} | {r['path_length_m']:.2f} |")
    rows += ['', 'Camera model and computation checks:', '',
             '| Stage | Field | Camera outputs checked | Max mean / R difference | Frame age median [sim s] | Detector inference median [wall ms] |',
             '|---|---|---:|---:|---:|---:|']
    for label, values in groups:
        for r in values:
            if 'camera_model_audit' not in r:continue
            a=r['camera_model_audit']
            rows.append(f"| {label} | {nav.NAMES[r['arm']]} | {a['observations_checked']} | "
                f"{a['maximum_mean_difference_m']:g} m / {a['maximum_R_difference_m2']:g} m² | "
                f"{r['frame_age_publish_median_sim_s']:.3f} | {r['inference_median_wall_ms']:.0f} |")
    rows += ['', 'What this evidence supports:', '',
        '- The camera mean/R interface matches the frozen residual-calibration model in actual runs.',
        '- Recorded camera-gap and command/state transaction defects can dominate the sensor-model comparison. '
        'The blind-turn mechanism is independently reproducible without a camera-noise assumption.',
        '- Planar aggregate error/coverage can conceal terminal heading failure. Report full state and outcomes.',
        '- A feasible optimized mean path does not establish trackability under a different local gate. '
        'The final-stop diagnosis exposes the soft global / hard local standoff mismatch.',
        '- These trials do not establish a GP benefit, statistical significance, temporal independence '
        'or a calibrated future-belief model. The scored coverage values are diagnostics, not calibration proof.', '',
        'The six earlier covariance/subset replay drives remain separate evidence with their own selection. '
        'The accepted IWAI result remains its published record. For the thesis, use the network extension '
        'and commissioning analysis with this explicit runtime validation boundary; a separate ICRA claim '
        'still requires compatible fusion/forecast semantics and independent route-discriminating trials.', '',
        'Current decisions and exact next actions: [ICRA_STATUS.md](../../../../docs/ICRA_STATUS.md). '
        'Technical repairs and policy recommendations: [runtime_integrity_audit.md](../../../../docs/runtime_integrity_audit.md).', '']
    (out/'runtime_results.md').write_text('\n'.join(rows))

    # A compact full-state diagnostic highlights what XY-only tables conceal.
    nav.style()
    fig,axes=nav.plt.subplots(1,3,figsize=(11.5,4.1),sharey=True,layout='constrained')
    for ax,(label,values) in zip(axes,groups):
        for index,arm in enumerate(nav.ARMS):
            r=next(r for r in values if r['arm']==arm)
            if 'belief_heading_p95_deg' not in r:continue
            ax.plot([index,index],[max(.05,r['belief_heading_p95_deg']),max(.05,r['belief_heading_final_deg'])],
                    color=nav.COLORS[arm],lw=1.)
            ax.scatter(index,max(.05,r['belief_heading_p95_deg']),color=nav.COLORS[arm],s=50,marker='o')
            ax.scatter(index,max(.05,r['belief_heading_final_deg']),color=nav.COLORS[arm],s=55,marker='x')
            ax.annotate(r['status'],(index,max(.05,r['belief_heading_final_deg'])),xytext=(4,7),
                        textcoords='offset points',fontsize=8)
        ax.set(xticks=[0,1,2],xticklabels=['Uniform','Geometry','GP'],title=label,yscale='log',ylim=(.05,250),xlim=(-.45,2.55))
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Absolute heading error [deg, log]')
    axes[0].scatter([],[],marker='o',color='#435360',label='Within-run 95th percentile')
    axes[0].scatter([],[],marker='x',color='#435360',label='Final scored belief')
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Heading and termination expose failures hidden by typical position error\n'
                 'One run per field per stage; descriptive comparisons, unequal drive lengths',fontsize=11)
    nav.savefig(fig,out/'heading_and_outcomes')
    sources[str(Path(__file__).relative_to(nav.REPO))]=nav.digest(Path(__file__))
    nav.writejson(out/'report_manifest.json',dict(sources=sources,
        outputs={p.name:nav.digest(p) for p in [out/'runtime_results.md',
                 *[out/('heading_and_outcomes.'+k) for k in ('pdf','svg','png')]]}))
    print('\n'.join(rows))


if __name__=='__main__':main()
