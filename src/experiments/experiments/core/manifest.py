import os
from typing import Any, Dict, List, Optional

from unav_common import manifest as common_manifest


_GIT_PROVENANCE_CACHE: Dict[str, Dict[str, Any]] = {}


def create_run_dir(log_dir: str, prefix: str = 'experiment') -> Dict[str, str]:
    run_id = common_manifest.generate_run_id(prefix)
    run_dir = os.path.join(log_dir, run_id)
    common_manifest.ensure_dir(run_dir)
    return {'run_id': run_id, 'run_dir': run_dir}


def snapshot_configs(run_dir: str, paths: List[str]) -> Dict[str, Optional[str]]:
    snapshots: Dict[str, Optional[str]] = {}
    for path in paths:
        if not path:
            continue
        name = os.path.basename(path)
        snapshots[name] = common_manifest.snapshot_file(path, run_dir, dest_name=name)
    return snapshots


def write_manifest(run_dir: str, data: Dict[str, Any], repo_root: str) -> str:
    manifest = dict(data)
    root = os.path.realpath(repo_root)
    if root not in _GIT_PROVENANCE_CACHE:
        _GIT_PROVENANCE_CACHE[root] = common_manifest.git_provenance(root)
    manifest.update(_GIT_PROVENANCE_CACHE[root])
    return common_manifest.write_manifest(run_dir, manifest)
