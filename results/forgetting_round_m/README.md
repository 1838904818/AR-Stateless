# Same-budget forgetting variants (round M)

Development screen on UNSW-NB15 and CIC-IDS-2017, fresh seeds 60–61, at the 1% FPR target.
The attack-memory byte budget is the same as AR-Stateless (200 rows per seen class).

- **M1**: herding admission (iCaRL exemplar rule on input features).
- **M2**: herding with a consistent float16 representation (about 1.8× the rows in the same bytes).
- **M3**: soft exposure matching: every new-family row is kept, weighted so that the family's total weight equals its memory quota.

The rule was fixed before running: an arm advances only if, on the screen mean, UNSW forgetting
falls by at least 5 points and final BAcc drops by at most 1 point on both datasets.
**No arm advanced** (`summary.json`), so no confirmation was run. Herding increased forgetting;
M3 reduced UNSW forgetting by 2.7 points.

One file per cell with every default and calibrated confusion matrix. Provenance strings
(script names, local paths, revision identifiers) were removed; numbers are unchanged.
