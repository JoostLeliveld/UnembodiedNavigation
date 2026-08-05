#!/usr/bin/env python3
"""Regenerate the frozen-path golden literals in ``tests/planning/test_efe_hit_miss_mixture.py``.

The golden block pins the *precision-blend* EFE objective and gradient to the
last bit, because that path backs the published single-camera paper and the
locked ``honest_campaign_v1`` campaign. Running this script rewrites those
numbers, i.e. it MOVES FROZEN METHOD.

Do not run it to "fix a failing test". Run it only when the frozen path has been
deliberately changed and that change has been signed off. Normal workflow is:
the test fails -> you broke the frozen path -> revert.

    python3 experiments/efe_hit_miss_mixture/regenerate_golden.py          # print
    python3 experiments/efe_hit_miss_mixture/regenerate_golden.py --write  # patch

The harness (camera, fields, evaluation points, cases) is imported from the test
file itself, so generator and test can never drift apart.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_FILE = REPO_ROOT / "tests" / "planning" / "test_efe_hit_miss_mixture.py"

_SRC_PKG_DIRS = (
    "src/reliability",
    "src/unav_common",
    "src/experiments",
    "src/planning",
    "src/perception",
    "src/state",
    "src/sim",
)
for _rel in _SRC_PKG_DIRS:
    _p = REPO_ROOT / _rel
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_ASSIGN_RE = re.compile(
    r"^GOLDEN: dict\[str, list\[tuple\[str, list\[str\]\]\]\] = .*?(?=\n\n)",
    re.MULTILINE | re.DOTALL,
)


def _load_harness(source: str) -> dict:
    """Exec the test module with the golden block neutralised."""
    stub = "GOLDEN: dict[str, list[tuple[str, list[str]]]] = {}"
    patched = _ASSIGN_RE.sub(stub, source)
    if patched == source:
        patched = source.replace("__GOLDEN_PLACEHOLDER__", "{}")
    ns: dict = {"__name__": "_golden_harness", "__file__": str(TEST_FILE)}
    exec(compile(patched, str(TEST_FILE), "exec"), ns)
    return ns


def _render(ns: dict) -> str:
    lines = ["GOLDEN: dict[str, list[tuple[str, list[str]]]] = {"]
    for name, params, kwargs in ns["_golden_cases"]():
        if getattr(params, "use_hit_miss_mixture", False):
            raise SystemExit(f"case {name!r} is not on the frozen path")
        lines.append(f"    {name!r}: [")
        for val, grad in ns["_evaluate_case"](params, kwargs):
            lines.append(f"        ({float(val).hex()!r}, [")
            for g in grad:
                lines.append(f"            {float(g).hex()!r},")
            lines.append("        ]),")
        lines.append("    ],")
    lines.append("}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true", help="patch the test file in place")
    args = ap.parse_args()

    source = TEST_FILE.read_text()
    block = _render(_load_harness(source))

    if not args.write:
        print(block)
        return 0

    if "__GOLDEN_PLACEHOLDER__" in source:
        patched = source.replace(
            "GOLDEN: dict[str, list[tuple[str, list[str]]]] = __GOLDEN_PLACEHOLDER__",
            block,
        )
    else:
        patched, n = _ASSIGN_RE.subn(lambda _m: block, source)
        if n != 1:
            raise SystemExit(f"expected exactly one GOLDEN assignment, matched {n}")
    TEST_FILE.write_text(patched)
    print(f"wrote golden block to {TEST_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
