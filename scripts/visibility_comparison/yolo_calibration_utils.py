#!/usr/bin/env python3
"""Shared helpers for YOLO score calibration."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize_scalar


def clip_probability(values, eps: float = 1e-6) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), float(eps), 1.0 - float(eps))


def logit(values, eps: float = 1e-6) -> np.ndarray:
    probs = clip_probability(values, eps=eps)
    return np.log(probs / (1.0 - probs))


def sigmoid(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return 1.0 / (1.0 + np.exp(-values))


def apply_temperature_scaling(scores, temperature: float, eps: float = 1e-6) -> np.ndarray:
    logits = logit(scores, eps=eps)
    temp = max(float(temperature), eps)
    return clip_probability(sigmoid(logits / temp), eps=eps)


def fit_temperature_scaler(
    scores,
    labels,
    *,
    min_temperature: float = 0.05,
    max_temperature: float = 10.0,
    eps: float = 1e-6,
) -> dict[str, float]:
    scores = clip_probability(scores, eps=eps)
    labels = np.asarray(labels, dtype=float).reshape(-1)
    if scores.shape[0] != labels.shape[0] or scores.shape[0] == 0:
        raise RuntimeError('Temperature scaling requires non-empty score/label arrays of equal length')
    logits = logit(scores, eps=eps)

    def _neg_log_likelihood(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        calibrated = clip_probability(sigmoid(logits / temperature), eps=eps)
        return float(
            -np.mean(
                labels * np.log(calibrated)
                + (1.0 - labels) * np.log(1.0 - calibrated)
            )
        )

    result = minimize_scalar(
        _neg_log_likelihood,
        bounds=(math.log(float(min_temperature)), math.log(float(max_temperature))),
        method='bounded',
    )
    temperature = math.exp(float(result.x))
    calibrated = apply_temperature_scaling(scores, temperature, eps=eps)
    raw_nll = float(_neg_log_likelihood(0.0))
    calibrated_nll = float(_neg_log_likelihood(math.log(temperature)))
    raw_brier = float(np.mean(np.square(scores - labels)))
    calibrated_brier = float(np.mean(np.square(calibrated - labels)))
    return {
        'temperature': float(temperature),
        'raw_nll': raw_nll,
        'calibrated_nll': calibrated_nll,
        'raw_brier': raw_brier,
        'calibrated_brier': calibrated_brier,
        'optimizer_success': bool(result.success),
    }
