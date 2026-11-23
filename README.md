# AR-Stateless

Class-incremental network intrusion detection with asymmetric replay and stateless
retraining.

In deployment, benign traffic is produced all the time and can be collected again at
every update. The raw records of past attacks cannot. AR-Stateless treats the two
asymmetrically:

1. a fresh benign sample (50,000 rows) is drawn at every task;
2. old attack families are kept only through a fixed replay memory of
   200 × |seen classes| rows, the same budget every compared method gets;
3. the previous model is discarded and a gradient-boosted tree ensemble
   (scikit-learn `HistGradientBoostingClassifier`) is trained from scratch;
4. an alert is raised when one attack score exceeds a threshold set on a benign
   hold-out pool to a target false-positive rate (1% in the main results).

The control **R3-L** uses the same learner, budget, splits and calibration, but keeps
benign rows in memory (200 per class) and does not refresh them. A hybrid control,
**R3-H**, refreshes benign data but still keeps the benign share of the memory. R3-H
recovers almost all of the gap between R3-L and AR-Stateless, so the gain comes from
refreshing benign data rather than from giving its memory share to attacks.

The full method, protocol, comparators, results and limitations are in
[`docs/AR_Stateless_technical_report.pdf`](docs/AR_Stateless_technical_report.pdf).

## Main results

Final balanced accuracy at the 1% FPR target, mean over seeds 10–14:

| | UNSW-NB15 | CIC-IDS-2017 | CSE-CIC-IDS2018 |
|---|---|---|---|
| AR-Stateless | 65.4 | 98.3 | 81.7 |
| ER-Bal-TabT (Ariffin et al.) | 56.3 | 92.7 | incomplete (3 of 5 seeds) |
| R3-L | 45.8 | 92.9 | 79.7 |
| R3-H | 64.9 | 97.8 | 82.1 |
| Best of DER++, iCaRL, ECBRS, BiC | 15.4 | 35.9 | 34.6 |

Measured test FPR stays between 0.75% and 1.15% for AR-Stateless on these three
datasets. On NSL-KDD, whose test benign traffic is shifted, it does not; see the
report.

## Code

| Module | Responsibility |
|---|---|
| `ar_stateless/data.py` | Load the prepared arrays, check their SHA-256, define the task order |
| `ar_stateless/splits.py` | Permanent row roles in TRAIN: hold-outs, benign stopping subset, per-task benign draw |
| `ar_stateless/memory.py` | Memory quotas, training/stopping slot split, memory update |
| `ar_stateless/model.py` | Tree ensemble fitting, class scores, the attack score |
| `ar_stateless/calibration.py` | Threshold at a target FPR from benign hold-out scores |
| `ar_stateless/trainer.py` | Per-task loop for one seed; produces confusion matrices |
| `ar_stateless/metrics.py` | Final and incremental balanced accuracy, forgetting, FPR, macro-F1 |
| `ar_stateless/stats.py` | Paired one-sided t test, Holm correction, exact sign-flip test |
| `ar_stateless/__main__.py` | Command line for one dataset × method × seed |
| `summarize.py` | Means over seeds and the AR-Stateless vs R3-L tests |

```bash
pip install -e ".[test]"
pytest
python -m ar_stateless --dataset UNSW-NB15 --seed 10 --data-root DATA --out runs
python -m ar_stateless --dataset UNSW-NB15 --seed 10 --method r3-l --data-root DATA --out runs
python summarize.py runs --dataset UNSW-NB15 --seeds 10 11 12 13 14
```

`DATA/<dataset>/` holds `train_x.npy`, `train_y.npy`, `test_x.npy`, `test_y.npy` and
`data_quality.json`. The arrays are built from the official CSV releases as described
in Section 3 and Appendix A of the report; the preparation scripts are available on
request.

The results in `results/` were produced by our earlier experiment pipeline. Run on the
same machine, this package reproduces its thresholds, boosting iteration counts and
confusion matrices exactly, for AR-Stateless and R3-L, seeds 10–14, on all four
datasets.

Tested with Python 3.10, NumPy 2.2.6, scikit-learn 1.7.2 and PyTorch 2.6.0 (CPU).
scikit-learn 1.7 or later is required for early stopping on an explicit validation set.

## Results

Each `result.json` holds, for every task, the default and calibrated (1/2/5%)
confusion matrices, the calibration record and the checkpoint hash.

| Folder | Contents |
|---|---|
| `results/confirmation/` | AR-Stateless (`R3-A`) and R3-L, three datasets, seeds 10–14 |
| `results/er_bal_tabt/` | ER-Bal-TabT, UNSW-NB15 and CIC-IDS-2017 |
| `results/er_bal_tabt_cse2018_incomplete/` | ER-Bal-TabT on CSE-CIC-IDS2018, the three seeds that finished |
| `results/published_methods_s1.json` | DER++, iCaRL, ECBRS and BiC: per-seed values and paired tests |
| `results/nsl_kdd/` | All methods on NSL-KDD (diagnostic, not part of the registered comparison) |
| `results/benign_volume_and_hybrid/` | Benign volume 5k/10k/20k and the hybrid control R3-H |

## License

MIT.
