"""Make repeated inference on the same image return the same numbers.

Two runs of the same model on the same frame must agree bit for bit, otherwise
"the models disagree" and "the GPU disagreed with itself" are indistinguishable
and every downstream comparison is noise.

``CUBLAS_WORKSPACE_CONFIG`` has to be set before the CUDA context is created, so
it is set at import time rather than inside :func:`set_deterministic`. Importing
this module before torch touches the GPU is enough; the adapter package imports
it first for that reason.
"""

from __future__ import annotations

import hashlib
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

DEFAULT_SEED = 0


def set_deterministic(seed: int = DEFAULT_SEED) -> dict:
    """Seed everything and switch off the nondeterministic fast paths.

    ``warn_only=True`` on ``use_deterministic_algorithms`` is deliberate: a few
    ops in these networks have no deterministic CUDA kernel, and a hard error
    would simply make the model unusable. The acceptance test checks the outcome
    (two runs, identical arrays) rather than trusting the flag.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    # TF32 rounds matmuls differently depending on kernel selection; off keeps
    # results reproducible across batch sizes as well as across runs.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32": False,
    }


def array_fingerprint(array) -> str:
    """Stable hash of an array's exact bytes — the determinism check's yardstick."""
    import numpy as np

    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def file_sha256(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


__all__ = ["set_deterministic", "array_fingerprint", "file_sha256", "DEFAULT_SEED"]
