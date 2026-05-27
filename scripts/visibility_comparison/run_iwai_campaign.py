#!/usr/bin/env python3
"""Compatibility wrapper for the renamed campaign runner.

Prefer:

    python3 scripts/visibility_comparison/run_visibility_campaign.py \
        --config scripts/visibility_comparison/paper_campaign_config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from run_visibility_campaign import main  # noqa: E402


if __name__ == '__main__':
    raise SystemExit(main())
