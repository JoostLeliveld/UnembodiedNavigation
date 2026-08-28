#!/usr/bin/env python3
"""Score every arm on every route and build every storyline, in one pass.

    python3 experiments/fusion_on_fixed_routes/story/build_all.py [--task=NAME]

Per arm per route: numbers.json, the three drive figures, the two fusion-mechanism figures
and the along-the-drive figure. Then that route's cross-arm figures. Then the cross-ROUTE
confirmation figures, which need every route present.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
from score import FOLDER, TASKS   # noqa: E402


def run(script, *args):
    result = subprocess.run([sys.executable, str(script), *args],
                            capture_output=True, text=True)
    for line in (result.stdout or "").splitlines():
        print("   ", line)
    if result.returncode != 0:
        tail = (result.stderr or "").strip().splitlines()[-3:]
        print(f"    FAILED: {' | '.join(tail)}")
    return result.returncode == 0


def main() -> int:
    args = sys.argv[1:]
    tasks = [a.split("=", 1)[1] for a in args if a.startswith("--task=")] or list(TASKS)
    ok = True
    for task in tasks:
        print(f"== {task}")
        for arm in FOLDER:
            ok &= run(HERE.parents[1] / "score.py", f"--task={task}", arm)
            ok &= run(HERE.parent / "arm.py", f"--task={task}", arm)
            ok &= run(HERE.parent / "fusion_examples.py", f"--task={task}", arm)
        ok &= run(HERE.parents[1] / "compare.py", f"--task={task}")
    if len(tasks) > 1:
        print("== across routes")
        ok &= run(HERE.parents[1] / "routes_compare.py")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
