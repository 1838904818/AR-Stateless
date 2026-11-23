"""The bounded replay memory.

The memory holds feature rows, not references into the data set: once a class is old,
the only rows of it that can ever be used again are the ones stored here.

Budget: ``200 x |seen classes|`` rows in total, the same for every method.

* ``attack_only_quotas`` (AR-Stateless) gives the whole budget to attack classes,
  split equally; the remainder goes one row each to the lowest class ids.
* ``per_class_quotas`` (the R3-L control) gives 200 rows to every seen class,
  Normal included, as in the published replay methods.

Each class's quota is split into training slots and early-stopping slots
(``slot_split``). When a quota shrinks, the class keeps a seeded subset of its own
stored rows; stored rows are never replaced with fresh rows of an old class.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .data import Dataset
from .splits import select_rows, task_seed

ROWS_PER_CLASS = 200


@dataclass
class Slot:
    train_ids: np.ndarray
    stop_ids: np.ndarray
    train_x: np.ndarray
    stop_x: np.ndarray


Memory = dict[int, Slot]


def attack_only_quotas(seen: list[int], benign: int) -> dict[int, int]:
    attacks = sorted(c for c in seen if c != benign)
    base, remainder = divmod(ROWS_PER_CLASS * len(seen), len(attacks))
    return {c: base + int(j < remainder) for j, c in enumerate(attacks)}


def per_class_quotas(seen: list[int]) -> dict[int, int]:
    return {c: ROWS_PER_CLASS for c in seen}


def slot_split(capacity: int) -> tuple[int, int]:
    """(training slots, early-stopping slots) for one class."""
    stop = max(5, math.ceil(capacity * 0.1))
    return max(0, capacity - stop), stop


def update_memory(data: Dataset, memory: Memory, quotas: dict[int, int],
                  new_pools: dict[int, tuple[np.ndarray, np.ndarray]],
                  seed: int, task: int) -> Memory:
    """Return the memory after ``task``.

    ``new_pools`` maps each class admitted at this task to its (training ids,
    early-stopping ids) currently available. Every other class in ``quotas`` must
    already be in ``memory`` and is shrunk from its own stored rows.
    """
    updated = {}
    for c, capacity in sorted(quotas.items()):
        n_train, n_stop = slot_split(capacity)
        rng_seed = task_seed(seed, task) * 10_007 + c
        if c in new_pools:
            train_pool, stop_pool = new_pools[c]
            train_ids = select_rows(train_pool, n_train, rng_seed)
            stop_ids = select_rows(stop_pool, n_stop, rng_seed + 53)
            slot = Slot(train_ids, stop_ids, data.x_train[train_ids], data.x_train[stop_ids])
        else:
            old = memory[c]
            keep_train = select_rows(np.arange(len(old.train_ids)), n_train, rng_seed)
            keep_stop = select_rows(np.arange(len(old.stop_ids)), n_stop, rng_seed + 53)
            slot = Slot(old.train_ids[keep_train], old.stop_ids[keep_stop],
                        old.train_x[keep_train], old.stop_x[keep_stop])
        if np.intersect1d(slot.train_ids, slot.stop_ids).size:
            raise AssertionError(f"Training and stopping slots overlap for class {c}")
        updated[c] = slot
    return updated


def memory_rows(memory: Memory) -> int:
    return sum(len(s.train_ids) + len(s.stop_ids) for s in memory.values())
