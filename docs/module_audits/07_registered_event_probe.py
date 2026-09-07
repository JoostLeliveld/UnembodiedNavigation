"""Reconstruct one manifest-bound fusion event; no accuracy or belief scoring.

Selection rule: P0 seed210 in the registered corrected-runtime selection, then
the earliest published batch containing at least two camera candidates.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"experiments/fusion_on_fixed_routes"))
import aligned
spec=importlib.util.spec_from_file_location("audit07_math",Path(__file__).with_name("07_fusion_probe.py"))
F=importlib.util.module_from_spec(spec); spec.loader.exec_module(F)
registry=json.loads((ROOT/"docs/localization_metrics_registry.json").read_text())
selection_path=ROOT/registry["network_navigation_runtime_pilot"]["selection"]
selection=json.loads(selection_path.read_text())
expected={"fusion_network_traverse__P0__seed210","fusion_network_traverse__P1__seed210",
          "fusion_network_traverse__P2__seed210"}
assert {r["key"] for r in selection["runs"]}==expected
assert len(selection["runs"])==3
entry=next(r for r in selection["runs"] if r["key"]=="fusion_network_traverse__P0__seed210")
run=ROOT/entry["run"]
hashes={}
for name in registry["required_runtime_evidence"]:
    digest=hashlib.sha256((run/name).read_bytes()).hexdigest()
    assert digest==entry["files"][name],name
    hashes[name]=digest
assert aligned.schema_version(run)>=4
groups={}
for row in aligned.observations(run):
    groups.setdefault(row["source_batch_id"],[]).append(row)
eligible=[(bid,rows) for bid,rows in groups.items() if len({r["camera"] for r in rows})>=2]
bid,rows=min(eligible,key=lambda pair:(pair[1][0]["fused_stamp"],pair[0]))
assert len(rows)==len({r["camera"] for r in rows}),"duplicate camera rows"
items=[F.obs("camera_"+r["camera"],r["aligned_xy"],r["aligned_cov"],t=r["common_capture_stamp"])
       for r in rows]
got=F.result(items,"joint_network",gate=.6)
head=rows[0]
np.testing.assert_allclose(got["mean"],[head["fused_x"],head["fused_y"]],atol=1e-12)
np.testing.assert_allclose(got["covariance"],head["fused_cov"],atol=1e-12)
assert set(got["used"])=={"camera_"+r["camera"] for r in rows if r["used"]}
outcomes=[r for r in aligned.assimilations(run) if r["source_batch_id"]==bid]
assert len(outcomes)==1
# Identity/timing only: do not turn NIS or a periodic belief into event accuracy.
outcome={k:v for k,v in outcomes[0].items() if k!="nis"}
out=dict(scope="One recorded fusion algebra/identity trace; no run accuracy, calibration or ranking",
    selection=str(selection_path.relative_to(ROOT)),selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
    run=entry["run"],input_hashes=hashes,batch=bid,frame="map_bev",units="m and m2",
    rows=[{k:r[k] for k in ["camera","source_batch_id","obs_stamp","obs_x","obs_y","cov",
        "common_capture_stamp","aligned_xy","aligned_cov","used","fused_stamp","decision_stamp"]} for r in rows],
    reconstructed=got,recorded_mean=[head["fused_x"],head["fused_y"]],recorded_covariance=head["fused_cov"],
    mean_max_difference_m=np.max(abs(np.array(got["mean"])-[head["fused_x"],head["fused_y"]])),
    covariance_max_difference_m2=np.max(abs(np.array(got["covariance"])-head["fused_cov"])),
    terminal=outcome)
print(json.dumps(F.jsonable(out),indent=2,allow_nan=False))
