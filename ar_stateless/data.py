"""Loading the prepared benchmark arrays.

Each dataset lives in its own folder containing four NumPy arrays and a receipt:

    train_x.npy  float32 (n_train, n_features)
    train_y.npy  int64   (n_train,)
    test_x.npy   float32 (n_test, n_features)
    test_y.npy   int64   (n_test,)
    data_quality.json    class order, task order, feature names, SHA-256 of each array

Labels are integer indices into ``class_order``; class 0 is always ``Normal``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATASET_FOLDERS = {
    "NSL-KDD": "nsl-kdd",
    "UNSW-NB15": "unsw-nb15",
    "CIC-IDS-2017": "cic-ids-2017",
    "CSE-CIC-IDS2018": "cic-ids-2018",
}
ARRAY_FILES = ("train_x.npy", "train_y.npy", "test_x.npy", "test_y.npy")
BENIGN = "Normal"


@dataclass(frozen=True)
class Dataset:
    name: str
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    class_order: list[str]
    tasks: list[list[int]]
    feature_names: list[str]

    @property
    def benign(self) -> int:
        return self.class_order.index(BENIGN)

    @property
    def attacks(self) -> list[int]:
        """Attack class ids in dataset order."""
        return [c for c in range(len(self.class_order)) if c != self.benign]

    def seen_after(self, task: int) -> list[int]:
        """Classes introduced in tasks 0..task, in introduction order."""
        return [c for part in self.tasks[: task + 1] for c in part]


def make_tasks(n_classes: int, classes_per_task: int) -> list[list[int]]:
    """Consecutive class ids per task; the first task always holds at least two classes."""
    tasks, start = [], 0
    while start < n_classes:
        size = max(classes_per_task, 2) if start == 0 else classes_per_task
        tasks.append(list(range(start, min(start + size, n_classes))))
        start += size
    return tasks


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_dataset(name: str, root: Path, *, verify_hashes: bool = True) -> Dataset:
    """Load a prepared dataset and check it against its receipt."""
    folder = Path(root) / DATASET_FOLDERS[name]
    receipt = json.loads((folder / "data_quality.json").read_text(encoding="utf-8"))
    if receipt.get("status") != "complete" or receipt.get("dataset") != name:
        raise ValueError(f"{folder} does not hold a complete copy of {name}")

    listed = {item["path"]: item for item in receipt["cache_files"]}
    arrays = {}
    for file in ARRAY_FILES:
        path, item = folder / file, listed[file]
        if verify_hashes and _sha256(path) != item["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        arrays[file] = np.load(path, mmap_mode="r", allow_pickle=False)

    classes_per_task = 1 if name == "NSL-KDD" else 2
    if receipt["tasks"] != make_tasks(len(receipt["class_order"]), classes_per_task):
        raise ValueError("Task order in the receipt differs from the protocol")
    if receipt["class_order"][0] != BENIGN:
        raise ValueError("Class 0 must be Normal")

    return Dataset(
        name=name,
        x_train=arrays["train_x.npy"],
        y_train=arrays["train_y.npy"],
        x_test=arrays["test_x.npy"],
        y_test=arrays["test_y.npy"],
        class_order=list(receipt["class_order"]),
        tasks=[list(t) for t in receipt["tasks"]],
        feature_names=list(receipt["feature_names"]),
    )
