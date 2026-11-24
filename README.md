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

A 15-page summary is in [`docs/AR_Stateless_summary.pdf`](docs/AR_Stateless_summary.pdf) (Chinese: [`docs/AR_Stateless_summary_zh.pdf`](docs/AR_Stateless_summary_zh.pdf)).
The full method, protocol, comparators, results and limitations are in
[`docs/AR_Stateless_technical_report.pdf`](docs/AR_Stateless_technical_report.pdf).

## Main results

Final balanced accuracy at the 1% FPR target, mean over seeds 10–14:

| | UNSW-NB15 | CIC-IDS-2017 | CSE-CIC-IDS2018 |
|---|---|---|---|
| AR-Stateless | 65.4 | 98.3 | 81.7 |
| ER-Bal-TabT (Ariffin et al.) | 56.3 | 92.7 | 80.8* |
| R3-L | 45.8 | 92.9 | 79.7 |
| R3-H | 64.9 | 97.8 | 82.1 |
| Best of DER++, iCaRL, ECBRS, BiC | 15.4 | 35.9 | 34.6 |

*On CSE-CIC-IDS2018 the difference from ER-Bal-TabT (+0.95 points, one-sided p = 0.155)
is not significant; this comparison was registered after three of its five seeds had been
seen, and two seeds were run on different hardware. On UNSW-NB15 and CIC-IDS-2017 the
differences are significant and every seed favours AR-Stateless.

Measured test FPR stays between 0.75% and 1.15% for AR-Stateless on these three
datasets. On NSL-KDD, whose test benign traffic is shifted, it does not; see the
report.

## Code

| Module | Responsibility |
|---|---|
| `ar_stateless/prepare.py` | Build the input arrays from the official CSV releases |
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
python -m ar_stateless prepare --dataset UNSW-NB15 --raw-root RAW --out DATA/unsw-nb15 --intermediate WORK
python -m ar_stateless --dataset UNSW-NB15 --seed 10 --data-root DATA --out runs
python -m ar_stateless --dataset UNSW-NB15 --seed 10 --method r3-l --data-root DATA --out runs
python summarize.py runs --dataset UNSW-NB15 --seeds 10 11 12 13 14
```

`DATA/<dataset>/` holds `train_x.npy`, `train_y.npy`, `test_x.npy`, `test_y.npy` and
`data_quality.json`. `prepare` builds them from the official CSV releases (RAW holds
the original dataset folders) as described in Section 3 of the report. UNSW-NB15 is
globally deduplicated and conflicting labels are removed before an 80/20 stratified
split; CIC-IDS-2017 and CSE-CIC-IDS2018 use a fixed group split with no identical
raw row in both training and test; NSL-KDD keeps its official split. Scaling and
encoding are fitted on the first task only. On 23 November 2026 the generated arrays
for all four datasets matched the arrays used for the published results
byte-for-byte (16 of 16 SHA-256 comparisons).

The results in `results/` were produced by our earlier experiment pipeline. Run on the
same machine, this package reproduces its thresholds, boosting iteration counts and
confusion matrices exactly, for AR-Stateless and R3-L, seeds 10–14, on all four
datasets.

Tested with Python 3.10, NumPy 2.2.6, scikit-learn 1.7.2 and PyTorch 2.6.0 (CPU).
scikit-learn 1.7 or later is required for early stopping on an explicit validation set.

## Results

Each `result.json` holds, for every task, the default and calibrated (1/2/5%)
confusion matrices, the calibration record and the checkpoint hash.
In the evidence folders below the results table, provenance strings (script names, local
paths) were replaced by neutral labels; every number and confusion matrix is unchanged,
and each folder's `SHA256SUMS` lists its files. Scheduler job numbers in those files are replaced by
`redacted-administrative-id`.

| Folder | Contents |
|---|---|
| `results/confirmation/` | AR-Stateless (`R3-A`) and R3-L, three datasets, seeds 10–14 |
| `results/er_bal_tabt/` | ER-Bal-TabT, UNSW-NB15 and CIC-IDS-2017 |
| `results/er_bal_tabt_cse2018_incomplete/` | ER-Bal-TabT on CSE-CIC-IDS2018, the three seeds that finished |
| `results/published_methods_s1.json` | DER++, iCaRL, ECBRS and BiC: per-seed values and paired tests |
| `results/published_methods_runs/` | DER++, iCaRL, ECBRS and BiC: every run, calibration record and the same-score re-scoring |
| `results/nsl_kdd/` | All methods on NSL-KDD (diagnostic, not part of the registered comparison) |
| `results/benign_volume_and_hybrid/` | Benign volume 5k/10k/20k and the hybrid control R3-H |
| `results/neural_replay_controls/` | Neural AR, Replay-BS, Freq and WACB on UNSW-NB15 and CIC-IDS-2017, seeds 10–14 |
| `results/forgetting_reduction_variants/` | Forgetting-reduction variants (capped new-class exposure, k-means memory): development screen and confirmation on seeds 15–19; none was confirmed |
| `results/forgetting_diagnostics_unsw/` | UNSW-NB15 forgetting diagnostics D1–D3 (offline joint model, matched exposure, larger memory), seeds 10–14 |
| `results/nsl_calibration_refresh/` | NSL-KDD calibration-pool refresh with UNSW-NB15 as control, seeds 10–14 |
| `results/implementation_equivalence/` | This package versus the experiment pipeline: 40 runs, all thresholds and confusion matrices identical (checksums in `SHA256SUMS`) |

## Use of AI tools

Claude (Anthropic) and Codex (OpenAI) were used to help implement and run the code
and to edit the documentation. I reviewed the results and the text and take full
responsibility for them.

## License

MIT.
