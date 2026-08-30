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
| 17 | 08-21 | CPU rank-blend of 10ep (0.910) + 6ep-holdout (0.900) distilled models → **LB 0.909** — blending in the weaker same-family member slightly dilutes the stronger one. Lesson: ensemble members must be comparable strength or genuinely diverse. Best remains 0.910. CPU-only inference of 2 models on 1,300 studies ran in ~2.5h (runtime datapoint) |

**Reconstruction note (08-30):** entries 18-23 below cover 08-23→08-26 Kaggle work that was submitted and scored but never logged/committed here — the session that ran it lost API access before it could pull results back and write to the repo. Recovered now from `kaggle competitions submissions` (authoritative for LB scores/dates) and each kernel's current `metrics.json`/log (soft-OOF); kernel slugs get overwritten on each push, so the code now under `kaggle/train-320*` etc. is the *final* version of each slug, not necessarily byte-identical to what produced every dated submission below. Distill targets throughout are `distill_targets_r2.csv` (round-2: retrain folds on round-1 distill targets → stronger teacher — the "distill round 2" item STRATEGY.md had queued next).

| 18 | 08-23 | 320px (up from 256px) all-data distilled EffV2-S, 24 slices, 10ep, mean-pool aggregation, distill-r2 targets → **LB 0.912** — resolution bump alone underperformed the 256px 0.910 baseline's next steps but confirmed the 320px cache/pipeline | | | **0.912** |
| 19 | 08-26 | Same 320px config, distill-r2 student retrained (`train-320s24`, soft-OOF 0.989) → **LB 0.915** — new best at the time | | | **0.915** |
| 20 | 08-26 | 320px 5-fold distill-r2 rank ensemble (`train-320-f0..f4`, per-fold soft-OOF 0.945-0.951, via `submit-ens-320`) → **LB 0.913** — *worse* than the single all-data model (0.915). Consistent with exp-17's lesson: at this label/arch maturity, ensembling near-identical single-seed members trades variance reduction for the all-data-training edge; the single model wins | | | 0.913 |
| 21 | 08-26 | `train-320-att`: BiGRU-over-slice-triplets + attention-MIL pooling (replaces mean-pool) + attention over the 4 series slots (replaces masked mean), same 320px/24sl/distill-r2 setup, soft-OOF 0.992 (architecture family later named **v2**) | | | not submitted alone |
| 22 | 08-26 | `train-v2`: production run of the v2 (BiGRU+attention) architecture, all-data, 320px, 24sl, distill-r2 targets, soft-OOF **0.992** | | | see #23 |
| 23 | 08-26 | v2 checkpoint submitted via `submit-v2` → **LB 0.922** — new overall best (+0.007 over exp-19), largest single jump since distillation itself. Aggregation (attention-MIL over slices+slots vs mean-pool) is the new highest-leverage lever, ahead of resolution | | | **0.922** |
| 24 | 08-29 | v2 architecture, 4/5 folds trained (`train-v2-f0..f3`; f4 not yet run), soft-OOF 0.940-0.944 per fold — 5-fold ensemble incomplete, not submitted; given exp-20's negative ensembling result at the plain-320 arch, a v2 5-fold ensemble is *not* pre-judged worth the remaining GPU-h without a CI-separated case for it | | | not submitted |
| 25 | 08-29 | v2 architecture at 384px (`train-v2-384`, all-data, 24sl, distill-r2, batch 1 vs 320's larger batch — same 10 epochs), soft-OOF **0.961** — *lower* than the 320px v2's 0.992 on the identical held-out val/targets, tracking behind at every epoch (ep9: 0.961 vs 0.992). Not the expected resolution win; batch-1 at 384px likely just needs more epochs/effective-batch tuning to converge as far in the same budget, so this is inconclusive rather than a negative result — **do not submit as-is**; either train longer or match effective batch size before trusting it against the 0.922 LB best | | | not submitted |
