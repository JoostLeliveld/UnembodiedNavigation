#!/usr/bin/env python3
"""Fail-fast host CUDA check: driver init, PyTorch discovery and real allocation."""
from __future__ import annotations

import ctypes
import json
import platform
import sys


def _cuda_error(cuda, result: int) -> dict[str, object]:
    name = ctypes.c_char_p()
    message = ctypes.c_char_p()
    cuda.cuGetErrorName(result, ctypes.byref(name))
    cuda.cuGetErrorString(result, ctypes.byref(message))
    return {
        'code': int(result),
        'name': name.value.decode() if name.value else 'unknown',
        'message': message.value.decode() if message.value else 'unknown',
    }


def main() -> int:
    report: dict[str, object] = {
        'python': platform.python_version(),
        'platform': platform.platform(),
    }
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        init_result = int(cuda.cuInit(0))
        report['driver_init'] = (
            {'code': 0, 'name': 'CUDA_SUCCESS'}
            if init_result == 0 else _cuda_error(cuda, init_result)
        )
    except Exception as exc:  # pragma: no cover - host diagnostic path
        report['driver_init'] = {'exception': f'{type(exc).__name__}: {exc}'}
        print(json.dumps(report, indent=2))
        return 1

    try:
        import torch

        report['torch'] = {
            'version': torch.__version__,
            'built_cuda': torch.version.cuda,
            'available': bool(torch.cuda.is_available()),
            'device_count': int(torch.cuda.device_count()),
        }
        if init_result == 0 and torch.cuda.is_available():
            sample = torch.arange(1024, dtype=torch.float32, device='cuda:0')
            report['torch'].update({
                'device_name': torch.cuda.get_device_name(0),
                'allocation_bytes': int(torch.cuda.memory_allocated(0)),
                'sample_sum': float(sample.sum().item()),
                'allocation_check': 'passed',
            })
    except Exception as exc:  # pragma: no cover - host diagnostic path
        report['torch_exception'] = f'{type(exc).__name__}: {exc}'

    print(json.dumps(report, indent=2))
    torch_report = report.get('torch', {})
    return int(
        init_result != 0
        or not isinstance(torch_report, dict)
        or torch_report.get('allocation_check') != 'passed'
    )


if __name__ == '__main__':
    sys.exit(main())
