"""The class-incremental run: one pass over the task sequence for one seed.

At every task ``t``:

1. Assemble training data: the new classes' full training pools, every old class's
   rows from memory, and Normal rows (a fresh 50,000-row draw for AR-Stateless, or
   the 180 stored rows for the R3-L control).
2. Fit a new tree model from scratch, early-stopped on the matching stopping rows.
3. Update the memory for the next task; old attack data become unavailable.
4. Calibrate thresholds on the benign calibration pool and evaluate on TEST.

``method="ar-stateless"`` is the proposed method; ``method="r3-l"`` keeps everything
identical except where benign data come from and how the memory is shared.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from . import model as tree
from .calibration import TARGET_FPRS, Threshold, calibrate
from .data import Dataset
from .memory import Memory, attack_only_quotas, memory_rows, per_class_quotas, update_memory
from .splits import benign_draw, make_roles, task_seed

METHODS = ("ar-stateless", "r3-l")
EVAL_BATCH = 4096


@dataclass
class TaskResult:
    task: int
    seen: list[int]
    training_rows: int
    stopping_rows: int
    memory_rows: int
    boosting_iterations: int
    fit_seconds: float
    thresholds: list[Threshold]
    calibrated_cm: dict[str, list[list[int]]]
    default_cm: list[list[int]]
    extra: dict = field(default_factory=dict)


def _labels(y: np.ndarray, benign: int, column: dict[int, int]) -> np.ndarray:
    """Dataset labels -> model columns (0 = Normal, k + 1 = k-th attack)."""
    return np.asarray([0 if c == benign else column[int(c)] + 1 for c in y], dtype=np.int64)


def _evaluate(fitted, data: Dataset, seen: list[int], thresholds: list[Threshold]):
    """Confusion matrices on TEST rows of seen classes, for each threshold and by argmax."""
    n, benign, attacks = len(data.class_order), data.benign, data.attacks
    seen_attack_columns = [attacks.index(c) for c in seen if c != benign]
    seen_attack_ids = np.asarray([attacks[k] for k in seen_attack_columns])
    to_dataset_order = [[benign, *attacks].index(c) for c in range(n)]

    rows = np.flatnonzero(np.isin(data.y_test, seen))
    calibrated = {t.target_fpr: np.zeros((n, n), dtype=np.int64) for t in thresholds}
    default = np.zeros((n, n), dtype=np.int64)
    for start in range(0, len(rows), EVAL_BATCH):
        batch = rows[start:start + EVAL_BATCH]
        x, y = np.asarray(data.x_test[batch]), np.asarray(data.y_test[batch])
        scores = tree.class_scores(fitted, x, n)
        score = tree.attack_score(scores)
        family = seen_attack_ids[scores[:, 1:][:, seen_attack_columns].argmax(axis=1)]
        for t in thresholds:
            predicted = np.where(score > t.value, family, benign)
            calibrated[t.target_fpr] += np.bincount(y * n + predicted, minlength=n * n).reshape(n, n)
        ordered = scores[:, to_dataset_order][:, seen]
        predicted = np.asarray(seen)[ordered.argmax(axis=1)]
        default += np.bincount(y * n + predicted, minlength=n * n).reshape(n, n)
    return calibrated, default


def run(data: Dataset, seed: int, method: str = "ar-stateless") -> list[TaskResult]:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    benign_memory = method == "r3-l"
    benign, attacks = data.benign, data.attacks
    column = {c: k for k, c in enumerate(attacks)}
    n_columns = len(data.class_order)
    roles = make_roles(data, seed)
    benign_scores_x = np.asarray(data.x_train[roles.calibration])

    memory: Memory = {}
    results = []
    for t, task in enumerate(data.tasks):
        seen = data.seen_after(t)
        new = [c for c in task if c != benign]

        # Rows available at this task: (training ids, stopping ids) per class.
        available = {c: (roles.pool[c], roles.holdout[c]) for c in new}
        if not benign_memory or t == 0:
            available[benign] = benign_draw(data, roles, seed, t)
        for c in seen:
            if c not in available:
                available[c] = (memory[c].train_ids, memory[c].stop_ids)

        train_ids = np.concatenate([available[benign][0], *[available[c][0] for c in seen if c != benign]])
        stop_ids = np.concatenate([available[c][1] for c in seen])
        if np.intersect1d(train_ids, stop_ids).size or np.isin(train_ids, roles.calibration).any() \
                or np.isin(stop_ids, roles.calibration).any():
            raise AssertionError("Training, stopping and calibration rows must be disjoint")

        started = time.perf_counter()
        fitted = tree.fit(data.x_train[train_ids], _labels(data.y_train[train_ids], benign, column),
                          data.x_train[stop_ids], _labels(data.y_train[stop_ids], benign, column),
                          random_state=task_seed(seed, t))
        fit_seconds = time.perf_counter() - started

        if benign_memory:
            admitted = new + ([benign] if t == 0 else [])
            quotas = per_class_quotas(seen)
        else:
            admitted, quotas = new, attack_only_quotas(seen, benign)
        memory = update_memory(data, memory, quotas, {c: available[c] for c in admitted}, seed, t)

        benign_scores = tree.attack_score(tree.class_scores(fitted, benign_scores_x, n_columns))
        thresholds = [calibrate(benign_scores, a) for a in TARGET_FPRS]
        calibrated, default = _evaluate(fitted, data, seen, thresholds)

        results.append(TaskResult(
            task=t, seen=seen, training_rows=len(train_ids), stopping_rows=len(stop_ids),
            memory_rows=memory_rows(memory), boosting_iterations=int(fitted.n_iter_),
            fit_seconds=fit_seconds, thresholds=thresholds,
            calibrated_cm={str(a): cm.tolist() for a, cm in calibrated.items()},
            default_cm=default.tolist()))
    return results


def to_json(results: list[TaskResult]) -> list[dict]:
    return [asdict(r) for r in results]
