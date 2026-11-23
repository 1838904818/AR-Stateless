"""Paired comparisons across seeds.

Each comparison pairs AR-Stateless with one other method on the same seeds and tests
the one-sided hypothesis that AR-Stateless is better:

* paired one-sided Student t test (df = n - 1);
* Holm's step-down correction over a family of such tests;
* the exact sign-flip test as a distribution-free check. With n pairs its smallest
  attainable p-value is 2**-n (1/32 for five seeds).
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy import stats


def paired_t(ours: list[float], theirs: list[float]) -> tuple[float, float]:
    """(t statistic, one-sided p) for H1: mean(ours - theirs) > 0."""
    result = stats.ttest_rel(ours, theirs, alternative="greater")
    return float(result.statistic), float(result.pvalue)


def sign_flip(ours: list[float], theirs: list[float]) -> float:
    """Exact one-sided sign-flip p-value for the mean paired difference."""
    d = np.asarray(ours, dtype=float) - np.asarray(theirs, dtype=float)
    observed = d.mean()
    flips = np.array(list(itertools.product((1, -1), repeat=len(d))))
    return float(np.mean((flips * d).mean(axis=1) >= observed - 1e-12))


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()
