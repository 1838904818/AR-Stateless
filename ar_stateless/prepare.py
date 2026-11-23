"""Reproduce the published input arrays from official CSV releases.

The transformation and ordering are fixed; test data never fit preprocessing.
Existing caches are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import DATASET_FOLDERS, make_tasks

NSL_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

NSL_ATTACK_FAMILY = {
    "normal": "Normal",
    "back": "DoS",
    "land": "DoS",
    "neptune": "DoS",
    "pod": "DoS",
    "smurf": "DoS",
    "teardrop": "DoS",
    "apache2": "DoS",
    "udpstorm": "DoS",
    "processtable": "DoS",
    "worm": "DoS",
    "mailbomb": "DoS",
    "satan": "Probe",
    "ipsweep": "Probe",
    "nmap": "Probe",
    "portsweep": "Probe",
    "mscan": "Probe",
    "saint": "Probe",
    "guess_passwd": "R2L",
    "ftp_write": "R2L",
    "imap": "R2L",
    "phf": "R2L",
    "multihop": "R2L",
    "warezmaster": "R2L",
    "warezclient": "R2L",
    "spy": "R2L",
    "xlock": "R2L",
    "xsnoop": "R2L",
    "snmpguess": "R2L",
    "snmpgetattack": "R2L",
    "httptunnel": "R2L",
    "sendmail": "R2L",
    "named": "R2L",
    "buffer_overflow": "U2R",
    "loadmodule": "U2R",
    "rootkit": "U2R",
    "perl": "U2R",
    "sqlattack": "U2R",
    "xterm": "U2R",
    "ps": "U2R",
}


def _load_raw(
    dataset: str, data_root: Path
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], list[str], list[str]]:
    if dataset == "NSL-KDD":
        train_path = data_root / "nsl-kdd" / "KDDTrain+.txt"
        test_path = data_root / "nsl-kdd" / "KDDTest+.txt"
        train = pd.read_csv(train_path, header=None, names=NSL_COLUMNS)
        test = pd.read_csv(test_path, header=None, names=NSL_COLUMNS)
        for frame in (train, test):
            frame["family"] = frame["label"].astype(str).str.strip().map(NSL_ATTACK_FAMILY).fillna("R2L")
            frame.drop(columns=["label", "difficulty"], inplace=True)
        class_order = ["Normal", "DoS", "Probe", "R2L", "U2R"]
        categorical = ["protocol_type", "service", "flag"]
        source_files = [str(train_path), str(test_path)]
    elif dataset == "UNSW-NB15":
        train_path = data_root / "unsw-nb15" / "UNSW_NB15_training-set.csv"
        test_path = data_root / "unsw-nb15" / "UNSW_NB15_testing-set.csv"
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        for frame in (train, test):
            frame.drop(columns=[c for c in ("id", "label") if c in frame], inplace=True)
            frame["family"] = frame.pop("attack_cat").astype(str).str.strip()
        class_order = [
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
        categorical = ["proto", "service", "state"]
        source_files = [str(train_path), str(test_path)]
    else:
        raise ValueError(f"unsupported dataset: {dataset}")

    cls_to_idx = {name: index for index, name in enumerate(class_order)}
    unknown_train = sorted(set(train["family"]) - set(cls_to_idx))
    unknown_test = sorted(set(test["family"]) - set(cls_to_idx))
    if unknown_train or unknown_test:
        raise ValueError(f"unknown labels: train={unknown_train}, test={unknown_test}")
    y_train = train.pop("family").map(cls_to_idx).to_numpy(dtype=np.int64)
    y_test = test.pop("family").map(cls_to_idx).to_numpy(dtype=np.int64)
    return train, y_train, test, y_test, class_order, categorical, source_files


def exact_feature_groups(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Return exact equality groups without relying on unverified row hashes."""
    if list(train.columns) != list(test.columns):
        raise ValueError("train/test feature columns differ")
    if train.dtypes.astype(str).to_dict() != test.dtypes.astype(str).to_dict():
        raise ValueError("train/test dtype mismatch requires explicit exact coercion review")
    combined = pd.concat([train, test], ignore_index=True)
    if combined.isna().any().any():
        raise ValueError("raw missing values need an explicit equality definition")
    for column in combined.select_dtypes(include=["floating"]).columns:
        values = combined[column].to_numpy(copy=True)
        values[values == 0] = 0.0
        combined[column] = values
    codes, unique = pd.MultiIndex.from_frame(combined).factorize(sort=False)
    if np.any(codes < 0):
        raise ValueError("unexpected null group")
    return codes.astype(np.int64), len(unique)


def conflict_flags(groups: np.ndarray, labels: np.ndarray, group_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Identify groups with >=2 distinct exact labels and all their member rows."""
    unique_pairs = pd.DataFrame({"group": groups, "label": labels}).drop_duplicates()
    counts = np.bincount(unique_pairs["group"].to_numpy(dtype=np.int64), minlength=group_count)
    conflicting = counts > 1
    return conflicting[groups], conflicting


def _transform(train, y_train, test, shifted, categorical, tasks):
    numeric = [c for c in train.columns if c not in categorical]
    frames = [f.copy() for f in (train, test, shifted)]
    for frame in frames:
        frame[numeric] = frame[numeric].replace([np.inf, -np.inf], np.nan)
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    transformer.fit(frames[0].loc[np.isin(y_train, tasks[0])])
    arrays = [np.asarray(transformer.transform(f), dtype=np.float32) for f in frames]
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Nonfinite transformed Exp H data")
    return (*arrays, list(transformer.get_feature_names_out()))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def counts(y, classes):
    return {name: int(np.count_nonzero(y == i)) for i, name in enumerate(classes)}


def clean_unsw(train, y_train, test, y_test):
    """Exact mixed-type raw tuples; drop ALL conflict members, keep one/group."""
    groups, n_groups = exact_feature_groups(train, test)
    labels = np.concatenate([y_train, y_test])
    bad_rows, bad_groups = conflict_flags(groups, labels, n_groups)
    _, first = np.unique(groups, return_index=True)
    keep = first[~bad_rows[first]]
    frame = pd.concat([train, test], ignore_index=True)
    selected_labels = labels[keep]
    train_ids, test_ids = train_test_split(
        keep, test_size=0.2, random_state=42, stratify=selected_labels, shuffle=True
    )
    # Sort by pooled source position, not by target label or difficulty.
    train_ids, test_ids = np.sort(train_ids), np.sort(test_ids)
    if len(np.intersect1d(groups[train_ids], groups[test_ids])):
        raise AssertionError("UNSW cleaned raw-feature groups overlap")
    quality = {
        "raw_rows": len(frame),
        "raw_feature_groups": n_groups,
        "conflict_groups_removed": int(bad_groups.sum()),
        "conflict_member_rows_removed": int(bad_rows.sum()),
        "same_label_excess_rows_removed": int(len(frame) - bad_rows.sum() - len(keep)),
        "retained_rows": len(keep),
        "retained_train_rows": len(train_ids),
        "retained_test_rows": len(test_ids),
        "train_test_exact_overlap_rows": 0,
        "remaining_conflict_groups": 0,
        "equality": (
            "All 42 raw model-input columns as exact parsed mixed-type tuples; "
            "excludes id, binary label, attack_cat; signed-zero normalized; "
            "no float32 conversion before grouping"
        ),
        "split": (
            "After full pooled official train+test cleaning: stratified 80/20, "
            "sklearn random_state=42; no oversampling or test caps"
        ),
        "raw_feature_dtypes": {k: str(v) for k, v in frame.dtypes.items()},
    }
    assert quality["raw_rows"] == (
        quality["conflict_member_rows_removed"]
        + quality["same_label_excess_rows_removed"]
        + quality["retained_rows"]
    )
    return (
        frame.iloc[train_ids].reset_index(drop=True),
        labels[train_ids],
        frame.iloc[test_ids].reset_index(drop=True),
        labels[test_ids],
        quality,
    )


def row_hashes(x, chunk_rows=25000):
    """Routing hashes only; candidates are always checked with complete bytes."""
    result = np.empty(len(x), dtype=np.uint64)
    for start in range(0, len(x), chunk_rows):
        values = np.array(x[start : start + chunk_rows], dtype="<f4", order="C")
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite numeric cache values")
        values[values == 0] = 0.0
        words = values.view("<u4")
        h = np.full(len(values), np.uint64(1469598103934665603))
        for j in range(words.shape[1]):
            h ^= words[:, j].astype(np.uint64)
            h *= np.uint64(1099511628211)
        result[start : start + len(values)] = h
    return result


def _row_bytes(row):
    row = np.array(row, dtype="<f4", copy=True)
    row[row == 0] = 0.0
    return row.tobytes()


def numeric_overlap(train, test, chunk_rows=25000):
    """Exact all-feature byte comparison, using hashes solely for candidates."""
    if train.shape[1] != test.shape[1]:
        raise ValueError("Incompatible feature widths")
    a, b = row_hashes(train, chunk_rows), row_hashes(test, chunk_rows)
    shared = np.intersect1d(a, b)
    ia = np.flatnonzero(np.isin(a, shared))
    ib = np.flatnonzero(np.isin(b, shared))
    keys = {_row_bytes(train[i]) for i in ia}
    matched = sum(_row_bytes(test[i]) in keys for i in ib)
    return {
        "test_matching_any_train_rows": int(matched),
        "test_rows": len(test),
        "test_matching_any_train_rate": float(matched / len(test)) if len(test) else None,
        "hash_candidate_train_rows": len(ia),
        "hash_candidate_test_rows": len(ib),
        "equality": (
            "Complete little-endian float32 feature bytes; signed-zero canonical; "
            "hash routing followed by full bytes, not hash-only equality"
        ),
    }


def save_dataset(
    out,
    dataset,
    x_train,
    y_train,
    x_test,
    y_test,
    classes,
    tasks,
    names,
    metadata,
    x_shift=None,
    y_shift=None,
):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "data_quality.json").exists():
        raise FileExistsError(f"Completed prepared cache already exists: {out}")
    arrays = {"train_x.npy": x_train, "train_y.npy": y_train, "test_x.npy": x_test, "test_y.npy": y_test}
    if x_shift is not None:
        arrays.update({"shift_x.npy": x_shift, "shift_y.npy": y_shift})
    receipts = []
    for filename, array in arrays.items():
        path = out / filename
        # Streaming CIC arrays may already be the final mmap file.
        if not isinstance(array, np.memmap) or Path(array.filename).resolve() != path.resolve():
            if path.exists():
                raise FileExistsError(path)
            np.save(path, array, allow_pickle=False)
        elif array.mode != "r":
            array.flush()
        receipts.append(
            {
                "path": filename,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        )
    metadata.update(
        {
            "status": "complete",
            "dataset": dataset,
            "class_order": classes,
            "tasks": tasks,
            "feature_names": list(names),
            "feature_count": len(names),
            "train_counts": counts(y_train, classes),
            "id_counts": counts(y_test, classes),
            "shift_counts": counts(y_shift, classes) if y_shift is not None else {},
            "cache_files": receipts,
        }
    )
    (out / "data_quality.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metadata


def prepare_tabular(dataset, raw_root, out, base):
    train, yt, test, ye, classes, categorical, sources = _load_raw(dataset, raw_root)
    original = {"train": counts(yt, classes), "test": counts(ye, classes)}
    if dataset == "UNSW-NB15":
        train, yt, test, ye, raw_quality = clean_unsw(train, yt, test, ye)
    else:
        groups, n_groups = exact_feature_groups(train, test)
        matches = np.isin(groups[len(train) :], groups[: len(train)])
        flags, bad = conflict_flags(groups, np.concatenate([yt, ye]), n_groups)
        raw_quality = {
            "raw_rows": len(train) + len(test),
            "raw_feature_groups": n_groups,
            "train_test_exact_overlap_rows": int(matches.sum()),
            "conflict_member_rows_retained": int(flags.sum()),
            "conflict_groups_retained": int(bad.sum()),
            "split": "Official KDDTrain+/KDDTest+ unchanged by published instruction",
            "equality": (
                "Exact parsed mixed-type feature tuples; signed-zero normalized; "
                "no float32 conversion before checking"
            ),
        }
    tasks = make_tasks(len(classes), 1 if dataset == "NSL-KDD" else 2)
    # _transform also applies the exact published feature order and one-hot policy.
    xt, xe, _, names = _transform(train, yt, test, test.iloc[:1], categorical, tasks)
    overlap = numeric_overlap(xt, xe)
    metadata = dict(
        base,
        original_counts=original,
        raw_feature_names=list(train.columns),
        source_files=[Path(p).name for p in sources],
        input_files=[
            {"path": Path(p).name, "bytes": Path(p).stat().st_size, "sha256": sha256(p)} for p in sources
        ],
        quality={"raw": raw_quality, "transformed": overlap},
        preprocessing=(
            "published ColumnTransformer: median-impute/StandardScaler numeric "
            "+ one-hot(handle_unknown=ignore), fitted only on Task0 train; final float32"
        ),
        train_cap_policy="none",
        test_cap_policy="none",
        caveats=[
            "Raw equality is defined before model float32 conversion; task0-only one-hot "
            "and float32 rounding may collapse distinct raw vectors. "
            "Transformed overlap is reported separately."
        ],
    )
    return save_dataset(out, dataset, xt, yt, xe, ye, classes, tasks, names, metadata)


def numeric_chunks(selection, indices, chunk_rows):
    for path, c, selected, n in selection:
        arr = np.load(path, mmap_mode="r", allow_pickle=False)
        for start in range(0, n, chunk_rows):
            rows = (
                arr[start : start + chunk_rows]
                if selected is None
                else arr[selected[start : start + chunk_rows]]
            )
            values = np.asarray(rows[:, indices], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite frozen cache {path}")
            yield values, c


def write_numeric_split(out, split, selection, indices, scaler, chunk_rows):
    n = sum(item[3] for item in selection)
    x = np.lib.format.open_memmap(
        out / f"{split}_x.npy", mode="w+", dtype=np.float32, shape=(n, len(indices))
    )
    y = np.lib.format.open_memmap(out / f"{split}_y.npy", mode="w+", dtype=np.int64, shape=(n,))
    cursor = 0
    for values, c in numeric_chunks(selection, indices, chunk_rows):
        z = scaler.transform(values).astype(np.float32)
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite transformed CIC rows")
        x[cursor : cursor + len(z)] = z
        y[cursor : cursor + len(z)] = c
        cursor += len(z)
    x.flush()
    y.flush()
    return x, y


def shard_selection(folder, classes, split):
    result = []
    for class_id, name in enumerate(classes):
        paths = sorted((folder / split / name).glob("part-*.npy"))
        if not paths:
            raise ValueError(f"Missing class: {split}/{name}")
        for path in paths:
            size = len(np.load(path, mmap_mode="r", allow_pickle=False))
            result.append((path, class_id, None, size))
    return result


def prepare_cic(dataset, raw_root, out, intermediate):
    from ._raw_cache.core import BuildOptions, build_dataset_cache

    slug = DATASET_FOLDERS[dataset]
    folder = intermediate / slug
    if not folder.exists():
        build_dataset_cache(slug, raw_root, intermediate, BuildOptions(chunk_rows=100000))
    else:
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if manifest["dataset"] != slug:
            raise ValueError("Existing intermediate dataset differs")
        expected = {entry["relative_path"] for entry in manifest["shards"]}
        actual = {path.relative_to(folder).as_posix() for path in folder.glob("*/*/part-*.npy")}
        if actual != expected or len(expected) != len(manifest["shards"]):
            raise ValueError("Existing intermediate shard inventory differs")
        for entry in manifest["shards"]:
            path = folder / entry["relative_path"]
            if path.stat().st_size != entry["size_bytes"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Existing intermediate shard differs: {entry['relative_path']}")
        for name, expected_hash in manifest["sidecars"].items():
            if sha256(folder / name) != expected_hash:
                raise ValueError(f"Existing intermediate sidecar differs: {name}")
    schema = json.loads((folder / "feature_schema.json").read_text(encoding="utf-8"))
    classes = schema["class_order"]
    original_names = schema["feature_columns"]
    names = [name for name in original_names if dataset != "CIC-IDS-2017" or name != "Fwd Header Length.1"]
    indices = [original_names.index(name) for name in names]
    train = shard_selection(folder, classes, "train")
    test = shard_selection(folder, classes, "test")
    tasks = make_tasks(len(classes), 2)
    scaler = StandardScaler()
    for values, _class_id in numeric_chunks([part for part in train if part[1] in tasks[0]], indices, 25000):
        scaler.partial_fit(values)
    out.mkdir(parents=True, exist_ok=False)
    xt, yt = write_numeric_split(out, "train", train, indices, scaler, 25000)
    xe, ye = write_numeric_split(out, "test", test, indices, scaler, 25000)
    metadata = {
        "preprocessing": "Task-0 TRAIN-only incremental float64 StandardScaler; float32 output",
        "split": "Frozen feature-hash group split, seed 42, natural class prevalence",
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "var": scaler.var_.tolist(),
            "n_samples_seen": int(scaler.n_samples_seen_),
        },
        "quality": {
            "raw_train_test_exact_overlap_rows": 0,
            "scope": "Identical feature hashes stay in one partition",
        },
    }
    return save_dataset(
        out, dataset, xt, yt, xe, ye, classes, tasks, ["numeric__" + name for name in names], metadata
    )


def prepare(dataset, raw_root, output, intermediate, reference=None):
    raw_root = Path(raw_root).resolve(strict=True)
    output = Path(output).resolve()
    intermediate = Path(intermediate).resolve()
    if output.exists() or output == raw_root or raw_root in output.parents:
        raise ValueError("Output must be new and outside the raw input directory")
    if dataset in ("CIC-IDS-2017", "CSE-CIC-IDS2018"):
        result = prepare_cic(dataset, raw_root, output, intermediate)
    else:
        result = prepare_tabular(dataset, raw_root, output, {})
    if reference is not None:
        expected = json.loads(
            (Path(reference) / DATASET_FOLDERS[dataset] / "data_quality.json").read_text(encoding="utf-8")
        )
        entries = {item["path"]: item for item in expected["cache_files"]}
        for item in result["cache_files"]:
            old = entries[item["path"]]
            for field in ("sha256", "bytes", "shape", "dtype"):
                if item[field] != old[field]:
                    raise ValueError(
                        f"First difference: {item['path']} {field}: {item[field]} != {old[field]}"
                    )
        for field in ("class_order", "tasks", "feature_names", "train_counts", "id_counts"):
            if result[field] != expected[field]:
                raise ValueError(f"First metadata difference: {field}")
    return result


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FOLDERS))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--intermediate", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    result = prepare(args.dataset, args.raw_root, args.out, args.intermediate, args.reference)
    print(json.dumps({"dataset": args.dataset, "arrays": result["cache_files"]}))


if __name__ == "__main__":
    cli()
