"""Read an R-sweep: rejection rate + belief accuracy per run.

Usage: python3 read_sweep.py logs/visibility_comparison/rsweep_r08 [more roots...]
Belief error comes from campaign_metrics (canonical columns); rejection stats
come from the pixel_corr_* diagnostics, which only exist on runs made after the
correction-chain consolidation.
"""
import csv, glob, math, sys, statistics as st
sys.path.insert(0, 'scripts/geometry_visibility')
import campaign_metrics as cm

def q(a, p): return sorted(a)[max(int(p*len(a))-1, 0)] if a else float('nan')

hdr = f"{'run':>14} {'corr':>6} {'reject%':>8} {'NISp50':>7} {'NISp95':>7} {'err p50':>9} {'err p95':>9}  reasons"
print(hdr); print('-'*len(hdr))
for root in sys.argv[1:]:
    for f in sorted(glob.glob(f'{root}/**/experiment.csv', recursive=True)):
        seen, reasons, nis = set(), {}, []
        for r in csv.DictReader(open(f)):
            k = (r.get('pixel_corr_apply_stamp') or '').strip()
            if k in ('', 'nan') or k in seen: continue
            seen.add(k)
            reasons[r['pixel_corr_reject_reason']] = reasons.get(r['pixel_corr_reject_reason'], 0) + 1
            try:
                v = float(r['pixel_corr_nis'])
                if math.isfinite(v): nis.append(v)
            except (ValueError, KeyError, TypeError): pass
        tot = sum(reasons.values())
        rej = 100*(tot - reasons.get('accepted', 0))/tot if tot else float('nan')
        try:
            run = cm.load_run(f)
            err = [float(e) for e in run['belief_error_m'] if e is not None and math.isfinite(float(e))]
        except Exception as exc:
            err = []; reasons = reasons or {'load_error': str(exc)[:40]}
        label = root.rstrip('/').split('/')[-1]
        note = reasons if tot else 'NO correction diagnostics (pre-consolidation run)'
        print(f"{label:>14} {tot:6d} {rej:7.1f}% {st.median(nis) if nis else float('nan'):7.2f} "
              f"{q(nis,.95):7.2f} {st.median(err) if err else float('nan'):8.3f}m {q(err,.95):8.3f}m  {note}")
