#!/usr/bin/env python3
from __future__ import annotations

import field_common as C


if __name__ == "__main__":
    manifest = C.freeze_inputs()
    print(C.OUT / "frozen/manifest.json")
    print(manifest["frozen_a3"]["sha256"])
