"""Command line: run one method on one dataset for one seed.

    python -m ar_stateless --dataset UNSW-NB15 --seed 10 --data-root DATA --out results/

Writes ``<out>/<dataset>/<method>_seed<seed>.json`` and prints the 1% FPR summary.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn

from . import __version__
from .data import DATASET_FOLDERS, load_dataset
from .metrics import trajectory
from .trainer import METHODS, run, to_json


def main() -> None:
    if sys.argv[1:2] == ["prepare"]:
        from .prepare import cli

        cli(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(
        prog="ar_stateless", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_FOLDERS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--method", default="ar-stateless", choices=METHODS)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    data = load_dataset(args.dataset, args.data_root)
    tasks = run(data, args.seed, args.method)

    out = args.out / DATASET_FOLDERS[args.dataset] / f"{args.method}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "method": args.method,
                "seed": args.seed,
                "class_order": data.class_order,
                "tasks": data.tasks,
                "versions": {
                    "ar_stateless": __version__,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "scikit-learn": sklearn.__version__,
                },
                "results": to_json(tasks),
            },
            indent=1,
        )
    )

    summary = trajectory([np.asarray(t.calibrated_cm["0.01"]) for t in tasks], data.tasks)
    print(
        f"{args.dataset} {args.method} seed {args.seed} at 1% FPR: "
        f"final BAcc {100 * summary['final_bacc']:.1f}%, "
        f"incremental BAcc {100 * summary['incremental_bacc']:.1f}%, "
        f"forgetting {100 * summary['forgetting']:.1f} pp -> {out}"
    )


if __name__ == "__main__":
    main()
