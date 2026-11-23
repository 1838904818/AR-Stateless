"""Row roles inside the TRAIN split.

Every TRAIN row of class ``c`` belongs to exactly one permanent role, fixed by the seed:

* ``holdout[c]``  10% of the class (at least 5 rows), never used for fitting.
  For Normal this is the calibration pool ``B_cal``; for attacks it is the
  early-stopping pool used while the class is new.
* ``pool[c]``     the remaining rows, from which training data are drawn.

Benign rows additionally have a fixed early-stopping subset: 10% of ``pool[Normal]``.
A per-task benign draw of 50,000 rows is split by membership in that subset, so a row
is either always a stopping row or never one, across all tasks.

The TEST split is never touched here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import Dataset

BENIGN_DRAW = 50_000


def select_rows(pool: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Deterministic sample of ``count`` ids from ``pool``, returned in ascending order."""
    order = np.random.default_rng(seed).permutation(len(pool))[:count]
    return np.sort(np.asarray(pool, dtype=np.int64)[order])


def task_seed(seed: int, task: int) -> int:
    return seed * 10_000 + task


@dataclass(frozen=True)
class Roles:
    pool: dict[int, np.ndarray]
    holdout: dict[int, np.ndarray]
    calibration: np.ndarray
    benign_stopping: np.ndarray


def make_roles(data: Dataset, seed: int) -> Roles:
    pool, holdout = {}, {}
    for c in range(len(data.class_order)):
        rows = np.flatnonzero(data.y_train == c)
        n_holdout = max(5, math.ceil(len(rows) * 0.1))
        if n_holdout >= len(rows):
            raise ValueError(f"Class {data.class_order[c]} is too small for a hold-out")
        shuffled = np.random.default_rng(seed * 100_003 + c * 1_009 + 17).permutation(rows)
        holdout[c], pool[c] = shuffled[:n_holdout], np.sort(shuffled[n_holdout:])

    b = data.benign
    n_stop = min(len(pool[b]) - 1, max(5, math.ceil(len(pool[b]) * 0.1)))
    benign_stopping = select_rows(pool[b], n_stop, seed * 100_003 + b * 1_009 + 7_919)
    return Roles(pool=pool, holdout=holdout, calibration=holdout[b].copy(),
                 benign_stopping=benign_stopping)


def benign_draw(data: Dataset, roles: Roles, seed: int, task: int) -> tuple[np.ndarray, np.ndarray]:
    """Fresh benign sample for one task, split into (training ids, stopping ids)."""
    b = data.benign
    drawn = select_rows(roles.pool[b], BENIGN_DRAW, task_seed(seed, task) * 10_007 + b)
    is_stop = np.isin(drawn, roles.benign_stopping)
    return drawn[~is_stop], drawn[is_stop]
