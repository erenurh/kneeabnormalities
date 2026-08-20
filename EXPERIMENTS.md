# Experiment log

| # | Date | Change (single variable where possible) | soft-OOF | gold-12 | Public LB |
|---|------|------------------------------------------|----------|---------|-----------|
| 1 | 08-19 | fold0 EffV2-S 256px 12sl 6ep, public hybrid labels | 0.834 | 0.871 | **0.872** |
| 2 | 08-19 | same, labels → v3c-blend rank-scale (BUG: uniform targets) | 0.821 | 0.816 | not submitted |
| 3 | 08-19 | same, labels → v3c-blend probability-scale | 0.849* | 0.869 | **0.880** |

| 4 | 08-19 | epochs 6→12 (same labels/config) | 0.841 (peak .8425 @ep9, ≤ 6ep's 0.849 — overfits soft labels past ~ep6) | 0.890* | not submitted |

| 5 | 08-20 | slices 12→16 (15 used, 5 triplets; batch 6) | **0.854** | 0.876 | candidate |
| 6 | 08-20 | laterality-aware aug: sagittal reversal + Med/Lat label swap | 0.835 ✗ | 0.851 | reverted — swap contradicts unmirrored cor/ax slots; coherent mirroring needs all planes flipped together |

*OOF measured against each run's own label set — not comparable across label versions; LB is the fair cross-run comparison.

Label sets vs gold-58 (ranking AUC): public best (hybrid_v4) 0.899 · our v3c alone 0.875 · v3c+2×hybrid blend 0.900-0.904.

Lessons: (1) label lift is real on hidden test (+0.008 LB, single-variable);
(2) training targets must be probability-scale, never percentile ranks;
(3) gold-12-per-fold is too small to rank models — use it only as a sanity anchor.
