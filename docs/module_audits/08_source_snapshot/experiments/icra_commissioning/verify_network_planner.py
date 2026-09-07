#!/usr/bin/env python3
"""Verify the selected planner artifacts/probe without re-fitting or claiming a drive."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src/planning'),str(REPO/'src/unav_common')]
from planning.core.camera_network import CameraNetworkModel


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root,probe_name,checks_path=None):
    manifest=json.loads((root/'manifest.json').read_text())
    checked={}
    def check(path,expected):
        actual=digest(path)
        if actual!=expected:raise RuntimeError(f'changed planner input or output: {path}')
        checked[str(path.relative_to(REPO))]=actual
    for p,h in manifest['sources'].items():check(REPO/p,h)
    for entry in manifest['artifacts'].values():
        path=REPO/entry['path'];check(path,entry['sha256']);CameraNetworkModel(path)
    probe=root/probe_name
    protocol=json.loads((probe/'protocol.json').read_text())
    results=json.loads((probe/'results.json').read_text())
    check(root/'manifest.json',protocol['network_manifest_sha256'])
    check(probe/'protocol.json',results['protocol_sha256'])
    for p,h in protocol['sources'].items():check(REPO/p,h)
    for p,h in results['files'].items():check(probe/p,h)
    status=dict(kind='planner_implementation_provenance_check',verified=True,
        outputs='fitted field and short optimization/reference figures; no robot execution',
        checks=checked,optimizer_converged={k:v['optimizer_success'] for k,v in results['results'].items()},
        remaining='full-route feasibility; shared live mean/R/fusion; held-out forecasts; matched navigation')
    if checks_path:
        status['test_transcript']=dict(text=checks_path.read_text(),sha256=digest(checks_path))
    (root/'verification.json').write_text(json.dumps(status,indent=2)+'\n')
    print(json.dumps(dict(verified=True,hashed_files=len(checked),optimizer_converged=status['optimizer_converged'])))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=REPO/'logs/studies/icra_commissioning_20260905/network_planner')
    p.add_argument('--probe',default='probe_reviewed')
    p.add_argument('--checks',type=Path)
    a=p.parse_args();verify(a.root.resolve(),a.probe,a.checks)
