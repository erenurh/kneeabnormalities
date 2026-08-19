# Operating rules (set by team lead)

- **NEVER submit to the competition without the user's explicit go-ahead for
  that specific submission.** Standing exception granted 2026-08-19: the
  model trained on our v3c-blend labels may be submitted once, without
  re-asking. Everything after that: ask first, every time.
- All data/GPU work runs on Kaggle via API-pushed kernels (`--accelerator
  NvidiaTeslaT4`; P100 is unusable with the preinstalled torch). Never
  download the DICOM dataset or bulk kernel outputs into the session
  container; only small CSVs/logs.
- Competition data (reports, labels, UIDs) never gets committed to this
  repo — it is public. Data-derived artifacts go to the private Kaggle
  dataset `seksenbes/rsna-knee-folds` or kernel outputs.
- Primary prize target: efficiency track (single model ≥0.92 in <15 min);
  main-LB ensemble is a derived bonus. Free GPU quota only (~30 h/week).
- Validation: dual-grouped frozen folds (`src/folds.py`), two-tier signal
  (soft-OOF + gold-58 anchor with bootstrap CIs, `src/validate.py`).
  Decisions require CI-separated wins; no micro-tuning on gold-58.
