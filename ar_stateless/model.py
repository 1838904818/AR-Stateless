"""The learner: one multiclass gradient-boosted tree ensemble per task.

Column 0 of the model is Normal; column ``k + 1`` is the ``k``-th attack class in
dataset order. A new model is fitted from scratch at every task, so nothing is
inherited from the previous one.

The attack score of a flow is

    s(x) = logsumexp_{a in attacks} z_a(x) - z_Normal(x),

where ``z`` are the model's raw class scores. Classes not yet trained are given a
score of -1e6, so they contribute nothing to the sum.

The score is computed with ``torch.logsumexp`` in float32 on the CPU, as in the
experiments. A NumPy implementation of the same formula rounds differently in the
last bit once there are two or more attack columns, which moves calibrated
thresholds by a few units in the last place.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

HYPERPARAMETERS = dict(
    max_iter=500,
    learning_rate=0.1,
    max_leaf_nodes=63,
    l2_regularization=1.0,
    class_weight="balanced",
    early_stopping=True,
)
ABSENT = -1e6


def fit(x: np.ndarray, y: np.ndarray, stop_x: np.ndarray, stop_y: np.ndarray,
        random_state: int) -> HistGradientBoostingClassifier:
    """Fit with early stopping on the given stopping rows (scikit-learn >= 1.7)."""
    if len(np.unique(y)) < 2:
        raise ValueError("The model needs at least two classes")
    model = HistGradientBoostingClassifier(random_state=random_state, **HYPERPARAMETERS)
    model.fit(x, y, X_val=stop_x, y_val=stop_y)
    return model


def class_scores(model: HistGradientBoostingClassifier, x: np.ndarray, n_columns: int) -> np.ndarray:
    """Raw scores for all ``n_columns`` model columns, float32, untrained columns at -1e6."""
    raw = model.decision_function(np.asarray(x))
    if raw.ndim == 1:  # binary model: split the single margin symmetrically
        raw = np.stack((-raw / 2, raw / 2), axis=1)
    scores = np.full((len(x), n_columns), ABSENT, dtype=np.float32)
    scores[:, model.classes_.astype(int)] = raw
    return scores


def attack_score(scores: np.ndarray) -> np.ndarray:
    """s(x) = logsumexp over attack columns minus the Normal column (float32)."""
    z = torch.from_numpy(np.ascontiguousarray(scores, dtype=np.float32))
    return (torch.logsumexp(z[:, 1:], dim=1) - z[:, 0]).numpy()
