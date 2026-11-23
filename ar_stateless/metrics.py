"""Metrics computed from per-task confusion matrices.

A confusion matrix ``cm[true, predicted]`` covers only TEST rows whose class has been
seen by that task. All metrics below take recalls over seen classes, Normal included.

* balanced accuracy (BAcc) at a task: mean recall over the classes seen so far;
* final BAcc: BAcc after the last task;
* incremental BAcc: mean of BAcc over all tasks;
* forgetting of a class: its best recall at any task before the last, minus its final
  recall (signed, not clipped). Reported as the mean over all classes introduced
  before the last task.
"""
from __future__ import annotations

import numpy as np


def recalls(cm: np.ndarray, seen: list[int]) -> dict[int, float]:
    cm = np.asarray(cm)
    return {c: cm[c, c] / cm[c].sum() for c in seen}


def balanced_accuracy(cm: np.ndarray, seen: list[int]) -> float:
    return float(np.mean(list(recalls(cm, seen).values())))


def false_positive_rate(cm: np.ndarray, benign: int) -> float:
    cm = np.asarray(cm)
    return float(1 - cm[benign, benign] / cm[benign].sum())


def macro_f1(cm: np.ndarray, seen: list[int]) -> float:
    cm = np.asarray(cm)
    scores = [2 * cm[c, c] / (cm[c].sum() + cm[:, c].sum()) for c in seen]
    return float(np.mean(scores))


def trajectory(cms: list[np.ndarray], tasks: list[list[int]]) -> dict[str, float]:
    """Final BAcc, incremental BAcc and mean forgetting for one run."""
    seen_by_task = [[c for part in tasks[: t + 1] for c in part] for t in range(len(tasks))]
    per_task = [recalls(cm, seen) for cm, seen in zip(cms, seen_by_task, strict=True)]
    old = [c for part in tasks[:-1] for c in part]
    forgetting = [max(r[c] for r in per_task[:-1] if c in r) - per_task[-1][c] for c in old]
    return {
        "final_bacc": float(np.mean(list(per_task[-1].values()))),
        "incremental_bacc": float(np.mean([np.mean(list(r.values())) for r in per_task])),
        "forgetting": float(np.mean(forgetting)),
    }
