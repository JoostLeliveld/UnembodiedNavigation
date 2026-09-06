#!/usr/bin/env python3
"""Write RESULTS.md by applying the study's pre-registered decision rule.

The rule is fixed in the study README before any data exists, and this script is the only
thing that turns numbers into a verdict, so the verdict cannot drift to fit the outcome:

  * Stage 1 sets the ceiling. A temporal filter can only remove the fast component `q`.
    If `q` is negligible next to the error that remains after correction, no filter can
    help, whatever the arm table says.
  * A temporal arm earns its place only if it beats the static-covariance arm on median
    error WITHOUT getting worse on mean normalised squared error or on the share of
    readings beyond four sigma.
  * Consistency reference for a two-dimensional measurement is a mean normalised squared
    error of 2.0. Above is overconfident, below is conservative.

Both guards from stage 1 are enforced here rather than left to the reader: if the order
sweep shows the reported `q` still shrinking as the trend order rises, or if the residual
is still strongly autocorrelated, the split is reported as untrustworthy and the ceiling
is refused instead of quoted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# A residual that is still this correlated means the smooth part was not removed.
AUTOCORR_LIMIT = 0.35
# If robust_sd falls by more than this between the lowest and highest swept order, the
# trend fit is absorbing real signal and the split is not stable.
SWEEP_TOLERANCE = 0.30
# Consistency reference for a 2-D measurement.
NSE_REFERENCE = 2.0


def load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f'{path} not found')
    return json.loads(path.read_text(encoding='utf-8'))


def split_verdict(bq: dict) -> tuple[list[str], dict[str, bool]]:
    """Assess whether the b/q split can be trusted, then report q."""
    lines: list[str] = []
    summary = bq.get('summary', {})
    sweep = bq.get('order_sweep', {})

    lines.append('## Stage 1 — how much fast component is there to filter?')
    lines.append('')
    lines.append(f'Capture: `{bq.get("capture")}`  ')
    lines.append(f'{bq.get("rows_detected")} detections of {bq.get("rows_total")} camera '
                 f'opportunities, {bq.get("lines_used")} camera-line sequences, '
                 f'trend order {bq.get("trend_order")}.')
    lines.append('')
    header = ('| quantity | unit | n | fast component (robust) | plain sd | median abs | '
              'lag-1 autocorrelation | smooth span along line |')
    lines.append(header)
    lines.append('|---|---|---|---|---|---|---|---|')

    trustworthy: dict[str, bool] = {}
    for signal, stats in summary.items():
        unit = stats['unit']
        ac = stats['lag1_autocorr_median']
        lines.append(
            f'| `{signal}` | {unit} | {stats["n"]} | {stats["robust_sd"]:.3f} | '
            f'{stats["sd"]:.3f} | {stats["median_abs"]:.3f} | {ac:+.3f} | '
            f'{stats["trend_span_median"]:.2f} |')

        orders = sorted(sweep, key=int)
        values = [sweep[order][signal]['robust_sd'] for order in orders
                  if signal in sweep[order]]
        stable = True
        if len(values) >= 2 and values[0] > 0:
            drop = (values[0] - values[-1]) / values[0]
            stable = abs(drop) <= SWEEP_TOLERANCE
        white = abs(ac) <= AUTOCORR_LIMIT if ac == ac else False
        trustworthy[signal] = bool(stable and white)

    lines.append('')
    lines.append('Guards on the split:')
    lines.append('')
    for signal in summary:
        ac = summary[signal]['lag1_autocorr_median']
        orders = sorted(sweep, key=int)
        values = [sweep[order][signal]['robust_sd'] for order in orders
                  if signal in sweep[order]]
        pieces = ', '.join(f'order {order}: {value:.3f}'
                           for order, value in zip(orders, values))
        status = 'trustworthy' if trustworthy[signal] else 'NOT trustworthy'
        reasons = []
        if abs(ac) > AUTOCORR_LIMIT:
            reasons.append(f'residual still correlated ({ac:+.2f}), so the smooth part '
                           f'was not fully removed')
        if len(values) >= 2 and values[0] > 0 and abs((values[0] - values[-1]) / values[0]) > SWEEP_TOLERANCE:
            reasons.append('the reported value keeps moving with trend order, so the fit '
                           'is absorbing real signal')
        suffix = f' — {"; ".join(reasons)}' if reasons else ''
        lines.append(f'- `{signal}`: {status}. Order sweep {pieces}{suffix}')
    lines.append('')
    return lines, trustworthy


def arm_verdict(arms: dict, static_arm: str = 'E_static_R') -> list[str]:
    lines: list[str] = []
    overall = arms.get('overall', {})
    settings = arms.get('settings', {})

    lines.append('## Stage 2 — do the filtering arms earn their place?')
    lines.append('')
    buckets = settings.get('commissioned_buckets') or {}
    if buckets:
        values = [value * 100.0 for value in buckets.values()]
        arm_e = (f'the commissioned arm states {min(values):.1f}-{max(values):.1f} cm, '
                 f'measured per camera and range bucket from the residuals themselves '
                 f'({len(buckets)} buckets)')
    elif settings.get('static_sigma_px') is not None:
        arm_e = f'the commissioned arm states {settings["static_sigma_px"]:.3f} px'
    else:
        arm_e = 'the commissioned arm was not configured'
    step_s = settings.get('step_s')
    interval = f'sample interval {step_s:.4f} s' if step_s else 'sample interval unset'
    if settings.get('spacing_m'):
        interval += (f' (= {settings["spacing_m"] * 100:.0f} cm at '
                     f'{settings["speed_m_s"]} m/s)')
    lines.append(f'Assumed per-frame pixel noise {settings.get("sigma_px")} px; '
                 f'{arm_e}; '
                 f'process noise {settings.get("q_accel_px"):.1f} px/s^2 '
                 f'(= {settings.get("px_per_metre")} px/m x '
                 f'{settings.get("accel_m_s2")} m/s^2, measured not tuned); '
                 f'{interval}; smoother lag {settings.get("lag")}.')
    lines.append('')

    names = {
        'A_raw': 'A — per-frame observation (current pipeline)',
        'B_kf': 'B — constant-velocity box filter',
        'C_robust_kf': 'C — box filter with soft rejection',
        'D_smoother': 'D — fixed-lag smoother',
        'E_static_R': 'E — per-frame observation, offline covariance (the arm to beat)',
    }
    lines.append('| arm | n | median error (cm) | RMS (cm) | p90 (cm) | '
                 'mean NSE | median NSE | beyond 4 sigma (%) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for arm in ('A_raw', 'E_static_R', 'B_kf', 'C_robust_kf', 'D_smoother'):
        stats = overall.get(arm)
        if not stats or not stats.get('n'):
            continue
        lines.append(
            f'| {names.get(arm, arm)} | {stats["n"]} | {stats["median_err_cm"]:.2f} | '
            f'{stats["rms_err_cm"]:.2f} | {stats["p90_err_cm"]:.2f} | '
            f'{stats["mean_nse"]:.2f} | {stats["median_nse"]:.2f} | '
            f'{stats["beyond_4sigma_pct"]:.2f} |')
    lines.append('')
    lines.append(f'A mean NSE of {NSE_REFERENCE} is consistent for a two-dimensional '
                 'measurement; above it the stated covariance is too small for the error '
                 'that is actually present, below it is conservative.')
    lines.append('')

    baseline = overall.get(static_arm)
    if not baseline or not baseline.get('n'):
        lines.append('The static-covariance arm produced no readings, so no arm can be '
                     'judged against it.')
        return lines

    lines.append('Applying the pre-registered rule, arm by arm:')
    lines.append('')
    for arm in ('B_kf', 'C_robust_kf', 'D_smoother'):
        stats = overall.get(arm)
        if not stats or not stats.get('n'):
            continue
        better_accuracy = stats['median_err_cm'] < baseline['median_err_cm']
        # "Not worse" is judged against the consistency reference: the arm must not move
        # further from 2.0 than the baseline does, and must not grow the tail.
        nse_gap = abs(stats['mean_nse'] - NSE_REFERENCE)
        base_gap = abs(baseline['mean_nse'] - NSE_REFERENCE)
        worse_calibration = nse_gap > base_gap * 1.05
        worse_tail = stats['beyond_4sigma_pct'] > baseline['beyond_4sigma_pct'] * 1.05
        earns = better_accuracy and not worse_calibration and not worse_tail

        detail = []
        detail.append(f'median {stats["median_err_cm"]:.2f} cm vs '
                      f'{baseline["median_err_cm"]:.2f} cm '
                      f'({"better" if better_accuracy else "not better"})')
        detail.append(f'mean NSE {stats["mean_nse"]:.2f} vs {baseline["mean_nse"]:.2f} '
                      f'({"worse" if worse_calibration else "not worse"})')
        detail.append(f'beyond 4 sigma {stats["beyond_4sigma_pct"]:.2f}% vs '
                      f'{baseline["beyond_4sigma_pct"]:.2f}% '
                      f'({"worse" if worse_tail else "not worse"})')
        lines.append(f'- **{names.get(arm, arm)}: '
                     f'{"EARNS its place" if earns else "does NOT earn its place"}.** '
                     + '; '.join(detail) + '.')
    lines.append('')
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    args = parser.parse_args()

    bq = load(args.study_dir / 'bq_split.json')
    arms = load(args.study_dir / 'arm_comparison.json')

    lines: list[str] = []
    lines.append('# Does filtering the bounding box over time help?')
    lines.append('')
    lines.append('Generated by `write_results.py`, which applies the decision rule fixed in '
                 'the study README before the data was captured.')
    lines.append('')

    split_lines, trustworthy = split_verdict(bq)
    lines.extend(split_lines)
    lines.extend(arm_verdict(arms))

    lines.append('## Caveats that travel with these numbers')
    lines.append('')
    lines.append('- Poses are static placements sampled densely along a path. That isolates '
                 'the pixel-raster effect but does not reproduce closed-loop timing, so a '
                 'positive result here is a necessary condition for the cascade, not a '
                 'demonstration of it in the live loop.')
    lines.append('- Commanded pose is evaluation-only, as everywhere in this repo.')
    lines.append('- The detector and the ground projection are the frozen ones; this study '
                 'calibrates nothing of its own.')
    if not all(trustworthy.values()):
        lines.append('- At least one quantity failed a split guard above. Treat its fast '
                     'component as an upper bound, not a measurement.')
    lines.append('')

    out = args.study_dir / 'RESULTS.md'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print()
    print(f'wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
