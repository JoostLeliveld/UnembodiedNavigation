"""Compatibility wrapper for legacy notebook helpers.

Use `scripts/legacy/botnav_efe_helpers.py` for direct edits.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_LEGACY_PATH = Path(__file__).with_name("legacy") / "botnav_efe_helpers.py"
_SPEC = spec_from_file_location("legacy_botnav_efe_helpers", _LEGACY_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load legacy helper module from {_LEGACY_PATH}")
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
    if _name.startswith("_"):
        continue
    globals()[_name] = getattr(_MODULE, _name)


if __name__ == '__main__':
    print('Deprecated entry path. Use scripts/legacy/botnav_efe_helpers.py')
