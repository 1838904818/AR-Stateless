# Confirmation of ×3 attack memory on unseen seeds

AR-Stateless with ×1 and ×3 attack memory on UNSW-NB15 and CIC-IDS-2017, ten seeds (75–84)
never used before (40 runs). Rule fixed before running: (a) UNSW forgetting lower with ×3,
one-sided paired t and exact sign-flip both p < 0.05; (b) mean reduction at least 5 points;
(c) final BAcc not lower by more than 1 point on either dataset.

| 1% FPR target | ×1 | ×3 |
|---|---|---|
| UNSW-NB15 final BAcc | 65.37 ± 0.77 | 67.71 ± 0.54 |
| UNSW-NB15 forgetting | 20.63 ± 1.10 | 13.80 ± 0.63 |
| CIC-IDS-2017 final BAcc | 98.95 ± 1.04 | 99.32 ± 0.89 |
| CIC-IDS-2017 forgetting | 0.34 ± 0.15 | 0.15 ± 0.04 |
| Replay size (MB, UNSW) | 0.53 | 1.34 |

UNSW forgetting fell on all ten seeds (paired t p = 9.3e-8, exact sign-flip p = 1/1024).
All three conditions hold: **confirmed**. This is a memory-capacity trade-off, not a new method.

Provenance strings were removed; numbers and confusion matrices are unchanged.
