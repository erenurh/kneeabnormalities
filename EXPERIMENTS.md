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
| 10 | 08-20 | Labeler v4 iterations (smoke-only, ~0.6h GPU): v4 (PF OA/MCL/synovitis/fracture-negation anchors) 0.870 — fracture negation hurt Fracture (.765) and Contusion; v4b (negation reverted) 0.8749 ≈ v3c 0.8748; blend contribution +0.001-0.002 (noise). Full run NOT launched per pre-committed gate; ~5h quota preserved. PF OA anchor (+.05) worth carrying into v5 with a stronger model when quota resets |
| 11 | 08-20 | Distillation: targets = 0.5*teacher-OOF(5-fold ens) + 0.5*v3c-blend (targets gold 0.9166). Student single model OOF-vs-labels **0.876** (+0.022 over 0.854), gold-12 0.884 — largest single-model gain; efficiency-entry candidate |
| 12 | 08-20 | Distilled single model → **LB 0.900** — new overall best; single model ≥ 5-fold ensemble (0.897). Efficiency-entry thesis validated: full knowledge transfer at 1/5 inference cost |
| 13 | 08-20 | All-data distilled single model (4,407 studies, same targets) → **LB 0.902** — new overall best; +0.002 over fold-holdout distilled (0.900), extra 25% data helped |
| 14 | 08-21 | 10-epoch all-data distilled model (curve rose through ep9) → **LB 0.910** — new overall best (+0.008 over 6ep; longer training pays on clean distill targets) |
| 15 | 08-21 | v3d silence imputation (synovitis<-effusion when unmentioned): HURTS (0.900→0.897-0.899) — our labeler already emits calibrated non-zero severities for silence; community gain only applies to hard-zero labelers. Not adopted. | | | |
| 16 | 08-21 | 320px/16sl cache building as 2 CPU half-kernels (zero GPU; measured: single kernel would be 21.1GB > 20GB output cap) — ready for Saturday's high-res training | | | |
| 18 | 08-23 | 320px all-data distilled 10ep (train-320; single variable vs exp-14: SIZE 256→320, batch 6→4, LR 2e-4). Ran 3.1h on T4 (well under cap). In-train fold-0 curve ep9 0.9918 vs 256px's 0.9939 — biased metric, does not rank generalization; fair readout needs LB. Submitted (user-approved) → **LB 0.912** — new overall best (+0.002 over 256px). Resolution pays on distill targets; 320px line becomes the efficiency-entry base. Open question for runtime week: 320px inference cost vs <15 min budget | | | **0.912** |
| 19 | 08-26 | Distill round 2: 5 folds retrained at 320px on r1 distill targets (per-fold OOF-vs-targets .945-.951); r2 targets = 0.5*fold-OOF + 0.5*v3c-blend → **gold-58 0.9237 vs r1's 0.9166 (+0.007, gate PASS)**. r2 student (320px, all-data, 10ep, 2.66h) → **LB 0.915** — new overall best (+0.003 over exp-18). Teacher-quality gains transfer; efficiency floor reached, 0.92 target 0.005 away | | | **0.915** |
| 17 | 08-21 | CPU rank-blend of 10ep (0.910) + 6ep-holdout (0.900) distilled models → **LB 0.909** — blending in the weaker same-family member slightly dilutes the stronger one. Lesson: ensemble members must be comparable strength or genuinely diverse. Best remains 0.910. CPU-only inference of 2 models on 1,300 studies ran in ~2.5h (runtime datapoint) |
