"""Summarise finished runs and test AR-Stateless against R3-L.

    python summarize.py results/ --dataset UNSW-NB15 --seeds 10 11 12 13 14

Reads the JSON files written by ``python -m ar_stateless`` and prints, at the 1% FPR
target, the mean final BAcc, incremental BAcc and forgetting for each method, then
the paired one-sided t test and exact sign-flip test on final BAcc.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ar_stateless.data import DATASET_FOLDERS
from ar_stateless.metrics import trajectory
from ar_stateless.stats import paired_t, sign_flip


def load(folder: Path, method: str, seed: int, target: str) -> dict[str, float]:
    run = json.loads((folder / f"{method}_seed{seed}.json").read_text())
    cms = [np.asarray(task["calibrated_cm"][target]) for task in run["results"]]
    return trajectory(cms, run["tasks"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results", type=Path)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FOLDERS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    parser.add_argument("--target", default="0.01")
    args = parser.parse_args()

    folder = args.results / DATASET_FOLDERS[args.dataset]
    runs = {m: [load(folder, m, s, args.target) for s in args.seeds] for m in ("ar-stateless", "r3-l")}
    for method, rows in runs.items():
        means = {k: 100 * np.mean([r[k] for r in rows]) for k in rows[0]}
        print(f"{method:13s} final BAcc {means['final_bacc']:5.1f}  "
              f"incremental BAcc {means['incremental_bacc']:5.1f}  forgetting {means['forgetting']:5.1f}")

    ours = [r["final_bacc"] for r in runs["ar-stateless"]]
    theirs = [r["final_bacc"] for r in runs["r3-l"]]
    t, p = paired_t(ours, theirs)
    print(f"final BAcc difference {100 * (np.mean(ours) - np.mean(theirs)):+.1f} pp, "
          f"t = {t:.2f}, one-sided p = {p:.3g}, sign-flip p = {sign_flip(ours, theirs):.4f}")


if __name__ == "__main__":
    main()
