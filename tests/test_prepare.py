"""Small CSV tests; real-data byte equivalence is a separate acceptance check."""

import json

import numpy as np
import pandas as pd
import pytest

from ar_stateless.prepare import NSL_COLUMNS, clean_unsw, prepare


def test_unsw_csv(tmp_path):
    raw = tmp_path / "raw"
    folder = raw / "unsw-nb15"
    folder.mkdir(parents=True)
    classes = [
        "Normal",
        "Generic",
        "Exploits",
        "Fuzzers",
        "DoS",
        "Reconnaissance",
        "Analysis",
        "Backdoor",
        "Shellcode",
        "Worms",
    ]
    frame = pd.DataFrame(
        [
            {
                "id": c * 20 + i,
                "label": int(c != 0),
                "attack_cat": name,
                "proto": "tcp",
                "service": "http",
                "state": "FIN",
                "dur": float(c * 100 + i),
                "bytes": i + 1,
            }
            for c, name in enumerate(classes)
            for i in range(20)
        ]
    )
    frame.iloc[::2].to_csv(folder / "UNSW_NB15_training-set.csv", index=False)
    frame.iloc[1::2].to_csv(folder / "UNSW_NB15_testing-set.csv", index=False)
    out = tmp_path / "output"
    result = prepare("UNSW-NB15", raw, out, tmp_path / "intermediate")
    assert result["class_order"] == classes
    assert np.load(out / "train_x.npy").dtype == np.float32
    assert sum(result["train_counts"].values()) == 160
    assert sum(result["id_counts"].values()) == 40
    assert result["quality"]["raw"]["remaining_conflict_groups"] == 0
    assert len(json.loads((out / "data_quality.json").read_text())["cache_files"]) == 4
    with pytest.raises(ValueError, match="Output must be new"):
        prepare("UNSW-NB15", raw, out, tmp_path / "another")


def test_nsl_csv(tmp_path):
    raw = tmp_path / "raw"
    folder = raw / "nsl-kdd"
    folder.mkdir(parents=True)
    records = []
    for i, label in enumerate(["normal", "neptune", "satan", "guess_passwd", "rootkit"]):
        row = dict.fromkeys(NSL_COLUMNS, 0)
        row.update(duration=i, protocol_type="tcp", service="http", flag="SF", label=label, difficulty=21)
        records.append(row)
    frame = pd.DataFrame(records, columns=NSL_COLUMNS)
    for filename in ("KDDTrain+.txt", "KDDTest+.txt"):
        frame.to_csv(folder / filename, index=False, header=False)
    result = prepare("NSL-KDD", raw, tmp_path / "output", tmp_path / "intermediate")
    assert result["tasks"] == [[0, 1], [2], [3], [4]]
    assert result["quality"]["raw"]["train_test_exact_overlap_rows"] == 5
    assert sum(result["train_counts"].values()) == 5


def test_unsw_conflicts_and_duplicates():
    train = pd.DataFrame({"value": [0.0, 0.0, 1.0, 1.0] + list(range(2, 22))})
    test = pd.DataFrame({"value": [0.0, 2.0]})
    labels = np.array([0, 1, 0, 0] + [i % 2 for i in range(2, 22)])
    *_arrays, quality = clean_unsw(train, labels, test, np.array([0, 0]))
    assert quality["conflict_member_rows_removed"] == 3
    assert quality["same_label_excess_rows_removed"] == 2


def test_cic_utf8_schema(tmp_path, monkeypatch):
    from ar_stateless._raw_cache import core

    def builder(dataset, raw, intermediate, options):
        folder = intermediate / dataset
        folder.mkdir(parents=True)
        schema = {"class_order": ["Normal", "DoS"], "feature_columns": ["flow–rate"]}
        (folder / "feature_schema.json").write_text(json.dumps(schema, ensure_ascii=False), encoding="utf-8")
        for split in ("train", "test"):
            for index, name in enumerate(schema["class_order"]):
                part = folder / split / name
                part.mkdir(parents=True)
                np.save(part / "part-00000.npy", np.array([[index], [index + 1]], dtype=np.float32))

    monkeypatch.setattr(core, "build_dataset_cache", builder)
    raw = tmp_path / "raw"
    raw.mkdir()
    result = prepare("CSE-CIC-IDS2018", raw, tmp_path / "out", tmp_path / "intermediate")
    assert result["feature_names"] == ["numeric__flow–rate"]
