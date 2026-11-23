"""Choosing the alert threshold from benign hold-out traffic.

For a target false-positive rate ``alpha`` and ``n`` benign calibration scores,
at most ``k = floor(alpha * n)`` of them may exceed the threshold. The threshold is
the ``(n - k)``-th smallest score, and a flow raises an alert only if its score is
strictly greater. Ties therefore never push the calibration FPR above ``alpha``.

Only benign TRAIN rows from the permanent calibration pool are used; test labels
play no part.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

TARGET_FPRS = (0.01, 0.02, 0.05)


@dataclass(frozen=True)
class Threshold:
    target_fpr: float
    value: float
    calibration_rows: int
    allowed_false_alarms: int
    false_alarms: int


def calibrate(benign_scores: np.ndarray, target_fpr: float) -> Threshold:
    n = len(benign_scores)
    if n == 0:
        raise ValueError("Empty calibration pool")
    allowed = int(Fraction(str(target_fpr)) * n)  # exact floor, no float rounding
    value = float(np.sort(benign_scores)[n - allowed - 1])
    false_alarms = int(np.count_nonzero(benign_scores > value))
    if false_alarms > allowed:
        raise AssertionError("Threshold exceeds the target on the calibration pool")
    return Threshold(target_fpr, value, n, allowed, false_alarms)
