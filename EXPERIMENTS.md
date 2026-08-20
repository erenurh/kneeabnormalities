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
| v8 | 08-20 | exp-5 reproduction for clean latest weights (deterministic: OOF 0.8539) — **morning submission candidate** |
| 7 | 08-20 | submit-kernel slice bug: 16sl model fed 12 slices → LB 0.875; fixed (15 slices, training-identical sampling) → **LB 0.881** (new best; OOF +0.005 → LB +0.001, within public-split noise) |
| 8 | 08-20 | folds 1-4 trained (same config as best) — per-fold OOF: .854/.840/.847/.850/.847; merged 4,407-study OOF vs gold-58 (full anchor, n=58): **0.857** CI(.818-.891). Weak columns: MCL .78, Synovitis .79, PF OA .80 |
| 9 | 08-20 | 5-fold rank ensemble (16sl effv2s, v3c-blend labels) → **LB 0.897** (+0.016 over single fold 0.881; above the +0.005-0.01 expectation) |
