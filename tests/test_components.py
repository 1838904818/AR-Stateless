import numpy as np
import pytest

from ar_stateless.calibration import calibrate
from ar_stateless.data import Dataset, make_tasks
from ar_stateless.memory import attack_only_quotas, per_class_quotas, slot_split
from ar_stateless.metrics import trajectory
from ar_stateless.splits import benign_draw, make_roles, select_rows
from ar_stateless.stats import holm, sign_flip
from ar_stateless.trainer import run


def toy_dataset(seed=0):
    rng = np.random.default_rng(seed)
    sizes = [3000, 300, 250, 200, 150, 120]
    centres = rng.normal(0, 1.5, size=(6, 6))
    centres[0] = 0

    def sample(scale):
        x = np.concatenate([rng.normal(centres[c], 1.0, size=(n // scale, 6)) for c, n in enumerate(sizes)])
        y = np.concatenate([np.full(n // scale, c) for c, n in enumerate(sizes)])
        return x.astype(np.float32), y.astype(np.int64)

    (xtr, ytr), (xte, yte) = sample(1), sample(3)
    return Dataset("toy", xtr, ytr, xte, yte, ["Normal", "A", "B", "C", "D", "E"],
                   make_tasks(6, 2), [f"f{i}" for i in range(6)])


def test_tasks():
    assert make_tasks(5, 1) == [[0, 1], [2], [3], [4]]
    assert make_tasks(10, 2) == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]


def test_quotas_spend_whole_budget_on_attacks():
    seen = [0, 1, 2, 3, 4, 5, 6]          # Normal + 6 attacks -> 1400 rows
    quotas = attack_only_quotas(seen, benign=0)
    assert 0 not in quotas
    assert sum(quotas.values()) == 1400
    assert sorted(quotas.values()) == [233] * 4 + [234] * 2
    assert sum(per_class_quotas(seen).values()) == 1400


def test_slot_split():
    assert slot_split(200) == (180, 20)
    assert slot_split(30) == (25, 5)


def test_select_rows_is_deterministic_and_sorted():
    pool = np.arange(100, 200)
    a, b = select_rows(pool, 10, 7), select_rows(pool, 10, 7)
    assert np.array_equal(a, b) and np.all(np.diff(a) > 0) and np.isin(a, pool).all()


def test_roles_are_disjoint():
    data = toy_dataset()
    roles = make_roles(data, seed=10)
    for c in range(6):
        assert not np.intersect1d(roles.pool[c], roles.holdout[c]).size
    train, stop = benign_draw(data, roles, seed=10, task=0)
    assert not np.intersect1d(train, roles.calibration).size
    assert not np.intersect1d(stop, roles.calibration).size
    assert np.isin(stop, roles.benign_stopping).all()


def test_calibration_never_exceeds_target():
    scores = np.r_[np.zeros(950), np.ones(50)].astype(np.float32)  # heavy ties
    th = calibrate(scores, 0.01)
    assert th.allowed_false_alarms == 10
    assert th.false_alarms <= 10


def test_trajectory_definitions():
    tasks = [[0, 1], [2]]
    cm0 = np.array([[9, 1, 0], [2, 8, 0], [0, 0, 0]])
    cm1 = np.array([[8, 1, 1], [5, 5, 0], [0, 1, 9]])
    out = trajectory([cm0, cm1], tasks)
    assert out["final_bacc"] == pytest.approx((0.8 + 0.5 + 0.9) / 3)
    assert out["incremental_bacc"] == pytest.approx(((0.9 + 0.8) / 2 + (0.8 + 0.5 + 0.9) / 3) / 2)
    assert out["forgetting"] == pytest.approx(((0.9 - 0.8) + (0.8 - 0.5)) / 2)


def test_statistics():
    assert sign_flip([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]) == pytest.approx(1 / 32)
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


@pytest.mark.parametrize("method", ["ar-stateless", "r3-l"])
def test_end_to_end(method):
    data = toy_dataset()
    results = run(data, seed=10, method=method)
    assert [r.seen for r in results] == [[0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5]]
    assert results[-1].memory_rows <= 200 * 6
    for r in results:
        th = next(t for t in r.thresholds if t.target_fpr == 0.01)
        assert th.false_alarms <= th.allowed_false_alarms
    assert run(data, seed=10, method=method)[-1].calibrated_cm == results[-1].calibrated_cm
