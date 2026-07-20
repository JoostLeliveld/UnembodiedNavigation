import json
import os
import shutil
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional


def generate_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def git_hash(repo_root: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, PermissionError, OSError, subprocess.CalledProcessError):
        return None


def snapshot_file(src_path: str, dest_dir: str, dest_name: Optional[str] = None) -> Optional[str]:
    if not src_path or not os.path.isfile(src_path):
        return None
    ensure_dir(dest_dir)
    name = dest_name if dest_name else os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, name)
    shutil.copyfile(src_path, dest_path)
    return dest_path


def write_manifest(dest_dir: str, data: Dict[str, Any], filename: str = "run_manifest.json") -> str:
    ensure_dir(dest_dir)
    path = os.path.join(dest_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    return path
