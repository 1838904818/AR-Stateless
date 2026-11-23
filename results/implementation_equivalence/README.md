# Implementation equivalence evidence

These 40 existing result files compare the standalone implementation with the
experiment implementation on the same prepared data, seed and machine. They cover
NSL-KDD, UNSW-NB15, CIC-IDS-2017 and CSE-CIC-IDS2018, two methods
(`ar-stateless` and `r3-l`), and seeds 10–14: ten files per dataset.

Each JSON contains the dataset, method, seed, standalone task outputs (`release`)
and experiment task outputs (`reference`). Across 170 paired task outputs, the
boosting iteration counts, thresholds at 1%, 2% and 5%, and all 680 paired
confusion matrices (default plus three calibrated views) are exactly equal.
The files are byte-for-byte copies of the completed comparison records; no
training or scoring was rerun to assemble this bundle. Superseded scoring
records are excluded.

`SHA256SUMS` lists the SHA-256 of every result JSON. It can be checked with
`sha256sum -c SHA256SUMS` from this directory. Relative checkpoint names in
the records identify original artifacts; checkpoint weights are not bundled.

This establishes same-machine implementation equivalence, not cross-platform
bitwise reproducibility, independent-data generalization, or a new performance
gain. It does not change any published experimental result.
