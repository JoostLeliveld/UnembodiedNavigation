"""Fit persistent error on seed 110, evaluate existing diagnostic seeds 111/112.

Same marginal constant R, mean, Q and logged observations in every arm. Only the
temporal decomposition changes. Hyperparameters use innovations, not truth error.
RGB was not recorded in these runs: this tests the temporal link independently.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import joblib
from scipy.stats import chi2
from replay import load, run_filter, OUT, REPO, ARTIFACT, CAPTURE, digest
from reliability.learned_box_correction import LearnedBoxCorrection
from derive_interpretations import camera_models


def _lag_correlation(values, lag):
    values = np.asarray(values, float)
    if lag <= 0 or len(values) <= lag:
        return np.nan
    left, right = values[:-lag], values[lag:]
    left, right = left-left.mean(), right-right.mean()
    scale = np.sqrt(np.sum(left*left)*np.sum(right*right))
    return float(np.sum(left*right)/scale) if scale > 0 else np.nan


def innovation_diagnostics(records):
    """Distribution and index-lag whiteness of normalized innovations.

    Temporal tests are performed separately per camera so simultaneous observations
    are never treated as a time series. Ljung-Box uses observation index lag; the
    logged cadence/gaps remain a limitation recorded in the protocol.
    """
    if not records:
        return {'n': 0}
    white=[]; nis=[]
    for row in records:
        covariance=np.asarray(row['innovation_covariance'],float)
        innovation=np.asarray(row['innovation'],float)
        white.append(np.linalg.solve(np.linalg.cholesky(covariance),innovation))
        nis.append(float(row['nis']))
    white=np.asarray(white);nis=np.asarray(nis)
    overall=dict(n=len(records),mean=white.mean(0).tolist(),
        covariance=np.cov(white.T,ddof=1).tolist() if len(white)>1 else None,
        mean_nis=float(nis.mean()),median_nis=float(np.median(nis)),
        above_chi2_95_fraction=float(np.mean(nis>chi2.ppf(.95,2))),
        above_chi2_99_fraction=float(np.mean(nis>chi2.ppf(.99,2))))
    per_camera={}
    for camera in sorted({row['camera'] for row in records}):
        chosen=[index for index,row in enumerate(records) if row['camera']==camera]
        values=white[chosen]; count=len(values); lags=min(10,max(0,count//5))
        axes={}
        for axis,name in enumerate(('x','y')):
            correlations=[_lag_correlation(values[:,axis],lag) for lag in range(1,lags+1)]
            valid=[value for value in correlations if np.isfinite(value)]
            statistic=(count*(count+2)*sum(value*value/(count-lag)
                       for lag,value in enumerate(correlations,start=1) if np.isfinite(value)))
            axes[name]=dict(lags=lags,autocorrelation=correlations,
                max_abs_autocorrelation=float(max(map(abs,valid))) if valid else None,
                ljung_box_statistic=float(statistic) if lags else None,
                ljung_box_p_value=float(chi2.sf(statistic,lags)) if lags else None)
        per_camera[camera]=dict(n=count,mean=values.mean(0).tolist(),
            covariance=np.cov(values.T,ddof=1).tolist() if count>1 else None,axes=axes)
    return dict(**overall,per_camera=per_camera,
        expected=dict(mean=[0.,0.],covariance=[[1.,0.],[0.,1.]],mean_nis=2.,
                      above_chi2_95_fraction=.05,above_chi2_99_fraction=.01))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-reference-gap-s', type=float)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    path = OUT/'thesis_evidence/selection.json'
    selection = json.loads(path.read_text())
    base = LearnedBoxCorrection(REPO/ARTIFACT)
    geometry = camera_models(json.loads((REPO/CAPTURE/'capture_manifest.json').read_text()),
                             capture_root=REPO/CAPTURE)
    models = joblib.load(OUT/'models.joblib')
    protocol = dict(status='existing_six_run_development_replay', selection_sha256=digest(path),
        max_reference_gap_s=args.max_reference_gap_s,
        fit_seed=110, evaluation_seeds=[111,112],
        selection='mean per-drive innovation Gaussian NLL; no GT tuning',
        fractions=[0., .25, .5, .75, .9], tau_s=[.5,2.,8.],
        limits=['capture-time idealization', 'logged preselected readings', 'no recorded driving RGB',
                'existing data already inspected', 'constant per-camera marginal R; no common bias fit'],
        sources={str(p.relative_to(REPO)):digest(p) for p in [Path(__file__),
          REPO/'experiments/icra_commissioning/replay.py',
          REPO/'src/reliability/reliability/correlated_camera.py']})
    (args.out/'protocol.json').write_text(json.dumps(protocol, indent=2))
    loaded = []
    for entry in selection['runs']:
        if entry['seed'] not in [110,111,112]:
            raise ValueError('unexpected selected seed')
        m,truth,odom,readings,_ = load(entry,base,geometry,
            max_reference_gap_s=args.max_reference_gap_s)
        loaded.append((entry,m,truth,odom,readings))
        print('Loaded', entry['task'],entry['seed'],len(readings),flush=True)

    def run(item, params):
        entry,m,truth,odom,readings = item
        cameras=sorted({r['camera'] for r in readings})
        return run_filter(m,truth,odom,readings,models,'constant',cameras,
                          correlated_parameters=params)

    grid = []
    for fraction in protocol['fractions']:
        for tau in ([2.] if fraction == 0 else protocol['tau_s']):
            params = dict(fraction=fraction,tau_s=tau)
            losses=[]
            for item in loaded:
                if item[0]['seed'] != 110:
                    continue
                _,_,innovations=run(item,params)
                losses.append(float(np.mean([.5*(np.linalg.slogdet(i['innovation_covariance'])[1]+
                    i['nis']+2*np.log(2*np.pi)) for i in innovations])))
            grid.append(dict(**params, fit_nll=float(np.mean(losses))))
            print('Fit',grid[-1],flush=True)
    best=min(grid,key=lambda p:p['fit_nll'])
    results=[]
    for item in loaded:
        if item[0]['seed']==110:
            continue
        arms={}
        for name,params in [('independent',None),('persistent',best)]:
            s,_,innovations=run(item,params)
            arms[name]=dict(**s,innovation_diagnostics=innovation_diagnostics(innovations))
        results.append(dict(task=item[0]['task'],seed=item[0]['seed'],arms=arms))
        print(results[-1],flush=True)
    (args.out/'results.json').write_text(json.dumps(dict(selected=best,grid=grid,
        runs=results,scope=protocol['status']),indent=2))


if __name__=='__main__':
    main()
