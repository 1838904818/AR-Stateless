# Memory size versus forgetting (curve)

AR-Stateless with attack memory ×0.5, ×1, ×1.5, ×2, ×3 and ×4 of 200 rows per seen class,
on UNSW-NB15 and CIC-IDS-2017, seeds 70–74 (60 runs). Everything else is unchanged.
Descriptive result; a fixed criterion (UNSW forgetting at least 5 points below ×1, final BAcc
not lower) selected ×3, which was then confirmed on unseen seeds (`../memory_confirmation/`).

`summary.json` holds means, standard deviations, replay size in MB (feature bytes and row IDs
only), paired t and exact sign-flip tests against ×1 with Holm correction, and where the errors
on old attack families go at the final task (Normal, another old family, a new family).
With five seeds the exact sign-flip test cannot fall below 0.156 after Holm correction.

Provenance strings were removed; numbers and confusion matrices are unchanged.
