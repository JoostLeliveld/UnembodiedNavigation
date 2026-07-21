#!/usr/bin/env python3
"""CANONICAL metric implementations for all analysis/study code.

WHY THIS EXISTS: a 2026-07-15 audit found ~15 hand-rolled copies of these
metrics across scripts/ with THREE different Spearman formulas (one with
broken tie handling), Brier with inconsistent clip epsilons, and two AUC
variants. Numbers computed from divergent copies are not comparable.

RULES
-----
- New analysis code MUST import from here:
      import sys; sys.path.insert(0, "scripts/shared")
      import metrics as M
- Do NOT re-implement any of these inline, even for a quick check.
- Existing finished studies keep their inline copies untouched (their numbers
  are published in VALIDATION/RESULTS docs; do not silently change them).
- For campaign-log LOADING (which columns are trustworthy), the canonical
  module is scripts/geometry_visibility/campaign_metrics.py — see the
  campaign-metrics skill. This file is only about scoring.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = [
    "sigmoid", "logit", "clip_prob", "probit_prob",
    "brier", "logloss", "auroc", "auprc", "ece", "fhtr", "spearman",
    "gaussian_nll_logit", "coverage_logit", "binned",
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -60.0, 60.0)))


def logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def clip_prob(p, eps=1e-4):
    return np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)


def probit_prob(mu_f, sigma_f):
    """MacKay probit approximation: E[sigmoid(f)] under N(mu_f, sigma_f^2)."""
    kappa = 1.0 / np.sqrt(1.0 + np.pi * np.asarray(sigma_f, dtype=float) ** 2 / 8.0)
    return sigmoid(kappa * np.asarray(mu_f, dtype=float))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def logloss(y, p, eps=1e-4):
    p = clip_prob(p, eps)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def auroc(y, p):
    """Mann-Whitney rank AUC with proper tie handling (NaN if one class)."""
    from scipy.stats import rankdata
    y = np.asarray(y, float); p = np.asarray(p, float)
    pos = y >= 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return math.nan
    r = rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auprc(y, p):
    """Average precision (area under the PR curve), sklearn step definition.

    AP = sum_k (R_k - R_{k-1}) * P_k over descending score thresholds, with tied
    scores consumed together. Returns NaN when there are no positive labels
    (recall is undefined), mirroring ``auroc``'s one-class behaviour. Preferred
    over AUROC when positives are rare, as in per-camera usable-observation
    labels.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    pos_total = float((y >= 0.5).sum())
    if pos_total == 0:
        return math.nan
    order = np.argsort(-p, kind="mergesort")  # stable sort, descending score
    y_sorted = (y[order] >= 0.5).astype(float)
    p_sorted = p[order]
    n = len(p_sorted)
    ap = 0.0
    tp = 0.0
    fp = 0.0
    prev_recall = 0.0
    i = 0
    while i < n:
        j = i
        while j < n and p_sorted[j] == p_sorted[i]:  # consume tied scores together
            if y_sorted[j] >= 0.5:
                tp += 1.0
            else:
                fp += 1.0
            j += 1
        precision = tp / (tp + fp)
        recall = tp / pos_total
        ap += (recall - prev_recall) * precision
        prev_recall = recall
        i = j
    return float(ap)


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    tot = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() == 0:
            continue
        tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def fhtr(y, p, tau_high=0.7, y_bad=0.5):
    """False-high-trust rate: P(predicted trust > tau_high | outcome bad)."""
    y = np.asarray(y, float); p = np.asarray(p, float)
    bad = y < y_bad
    if bad.sum() == 0:
        return math.nan
    return float(np.mean(p[bad] > tau_high))


def spearman(a, b):
    """Tie-correct Spearman rho via scipy. Returns (rho, n_finite)."""
    from scipy.stats import spearmanr
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return math.nan, int(m.sum())
    return float(spearmanr(a[m], b[m]).statistic), int(m.sum())


def gaussian_nll_logit(y, mu_f, sigma_f, noise_var=0.05, eps=1e-4):
    """NLL of logit-space targets under the predictive Gaussian (score targets)."""
    t = logit(clip_prob(y, eps))
    var = np.asarray(sigma_f, float) ** 2 + noise_var
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (t - np.asarray(mu_f, float)) ** 2 / var))


def coverage_logit(y, mu_f, sigma_f, noise_var=0.05, z=1.645, eps=1e-4):
    """Fraction of logit targets inside the central 90% predictive interval."""
    t = logit(clip_prob(y, eps))
    sd = np.sqrt(np.asarray(sigma_f, float) ** 2 + noise_var)
    return float(np.mean(np.abs(t - np.asarray(mu_f, float)) <= z * sd))


def binned(x, y, edges, fn=np.nanmedian):
    """Bin y by x; returns (bin_centers, fn per bin, counts)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    out, centers, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        centers.append(0.5 * (lo + hi)); ns.append(int(m.sum()))
        out.append(fn(y[m]) if m.sum() else math.nan)
    return np.array(centers), np.array(out), np.array(ns)
