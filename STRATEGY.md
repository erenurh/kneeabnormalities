# STRATEGY — RSNA Knee Abnormality Detection

Ranked plan, written 2026-08-19 (64 days to final submission). Premise from ANALYSIS.md: the metric is unweighted macro ROC-AUC; reports are train-only; label quality and validation trust — not architecture — decide placement.

**Progress (as of 2026-08-30):** LB 0.872 → **0.922** (see EXPERIMENTS.md #1-25; #18-25 were run 08-23→08-29 and only reconciled into this repo on 08-30 after a session gap with no Kaggle API access). Proven levers, in measured order of impact: attention-MIL aggregation (BiGRU over slice triplets + attention over series slots, replacing mean-pool) **+0.007 LB (new best lever)**, distillation from the 5-fold ensemble teacher (+0.02 single-model), longer training on clean distill targets (+0.008), fold ensembling on the plain-320 arch (+0.016, but *negative* at the v2/attention arch — see #20), own labels (+0.008), 320px resolution alone (no win over 256px without the arch change — see #18). Distill round 2 (retrain folds on round-1 distill targets → stronger teacher) shipped and is the label version behind #18-25. Closed thread: v2 arch at 384px (#25-26) — tried both plain batch-1 and a batch-matched (grad-accum) rerun, neither closed the gap to the 320px v2 run's soft-OOF at the same 10-epoch budget; deprioritized, not a submission candidate without a much bigger (unproven) epoch/LR investment. Current best remains the 320px v2 model (LB 0.922).

**Gold-58 anchor, run for the first time on the v2 line (#27, 08-31):** macro-AUC 0.9218, CI [0.892, 0.947] — matches the LB score (0.922) almost exactly, tighter than the old community CV→LB slope (LB≈0.825·gold+0.184) predicted. Read this as encouraging, not yet re-fit: n=58, CI is ±0.025-0.03, one point isn't enough to respec the slope. Confirms the same weak columns flagged back at exp-8/10 are still weak at the new best: **Synovitis (.839), Lateral OA (.845), PF OA (.876), Lateral Meniscus (.872)** — per §4's budget-allocation rule, these are exactly the model-limited/teacher-limited findings the roadmap says should get resolution or labeler iterations, and neither has happened since v3/v4 labels. Open thread: v2 5-fold ensemble is 4/5 trained (f4 missing) and deprioritized given #20's negative ensembling result on the sibling arch — next roadmap items: a per-finding push on the confirmed-weak four (labeler v5/Qwen3-14B queued since #10, or targeted resolution/pooling experiments per §4), and the efficiency-track runtime harness (never measured against the real 9h/15min budgets).

**Prize decision (2026-08-19, team lead):** primary target = **efficiency prize** (single model, ≥0.92 private AUC, <15 min, fp16, no TTA — the prize math trades 0.01 AUC ≈ 12 min, so accuracy-per-parameter is the whole game and the track is under-contested); secondary = main-LB top-10 as a bonus via a rank ensemble derived from the same line. Compute = free Kaggle quota only (~30 GPU-h/week ⇒ ~25 h/week on the single-model line, ~5 h/week on 1–2 diversity backbones for the ensemble). Final selections: submission 1 = efficiency single model, submission 2 = rank ensemble. The runtime harness is measured in the week of Sep 1, not at the end.

Operating model: all data/GPU work runs on Kaggle (API-pushed kernels, free GPU quota, checkpoint-resume across ≤12 h sessions); this repo holds code, configs, and pulled-back artifacts (OOF CSVs, audit JSON, weights). Nothing heavy is ever downloaded locally.

---

## 1. Weak label engine (priority #1 — the competition's actual axis)

Goal: turn 4,349 multilingual reports into **per-finding soft labels calibrated to the strict, image-derived gold definitions** ("on-the-fence = negative"). All of this is offline/train-time — it costs zero of the 9 h budget. Ceiling to beat: best public label set = 0.893 macro-AUC vs gold.

1. **Extraction:** translate-then-label as default (offline NMT or LLM translation → one strong English labeler), A/B'd against direct multilingual prompting on gold + hand-checked per-language samples. Labeler = open-weights LLM (Qwen-class) run on Kaggle GPU or locally-hosted; hosted APIs only if ruling #733965 verifies. Prompt asks per finding for a **structured triple: {mentioned?, severity grade in the report's own words, certainty}** — never a direct binary. The gold rubric's exact thresholds (ANALYSIS §1.2) are embedded in the prompt.
2. **Severity→label mapping fitted on gold-58, per finding:** the mapping from ("small effusion", "low-grade sprain", "intrasubstance signal") to 0/1 is a tiny supervised problem with 58 examples — fit with repeated stratified K-fold over the 58, report bootstrap CIs, and freeze early. This single step is where the ~82% report-gold gap is closed or lost.
3. **Ensemble decorrelated labelers:** our LLM triple-extractor + per-language regex/negation rules + ≥1 public label set (stevenleehans v4_blend). Community evidence: the weakest standalone labeler adds the most blend lift. **Disagreement = free per-sample noise estimate** → per-sample, per-finding confidence weights in the loss.
4. **Finding-specific silence handling:** "not addressed" imputed per finding from measured conditionals (Baker's silence ≈ negative; synovitis silence ≠ negative — impute from effusion), not blanket.
5. **Per-language/per-site QA:** macro-AUC of labels vs gold sliced by language; for languages with no gold coverage, hand-check 20 translated reports each (I draft, user or a clinician-friend spot-checks). Down-weight, don't drop, low-confidence languages.
6. Versioned outputs: `labels_vX.csv` (12 soft probs + 12 confidences per study) with the exact prompts/configs committed alongside. Every model run records which label version trained it.

## 2. Validation design (tied priority #1 — the highest-risk item)

With 58 gold studies and a 70% private LB, an untrustworthy CV kills us silently.

1. **Folds:** 5-fold, **study-level, dual-grouped** — union-find over (byte-identical/near-duplicate reports ∪ same site fingerprint) so the 183 duplicate-report studies and same-site studies never straddle folds. Device IDs are stripped from the DICOMs (measured), so the site fingerprint is (manufacturer, model, field strength) — 45 combos — optionally refined with report language. Stratify by a coarse label-burden bucket from soft labels. Fold assignment is versioned data, produced once by the audit kernel.
2. **Two-tier signal:**
   - *Primary (high-volume, biased):* OOF AUC against **soft report labels** on 4,349 studies — precise for ranking model variants trained on the same label version.
   - *Anchor (unbiased, tiny):* AUC against **gold-58**, always via models that never saw those studies (they stay in a permanent quarantine set, excluded from all training including labeler threshold fitting except via internal K-fold). Report with bootstrap CIs; expect ±0.02–0.04 noise; never chase single-run moves inside the CI.
3. **CV→LB tracking:** log (gold-CV, soft-OOF, public LB) for every submission; refit the slope (community fit: LB ≈ 0.825·gold + 0.184). Decisions use local CV; the public LB (30%) is only a sanity check — 5 subs/day are spent on calibration probes early, not late-stage thrash.
4. **Per-finding dashboard:** 12 per-finding AUCs on both tiers + model-limited vs teacher-limited classification per finding. This drives budget allocation (§4).
5. **Final selection rule (pre-committed now):** one max-robustness ensemble chosen by gold-CI-lower-bound, one efficiency entry. No last-week public-LB-chasing swaps.

## 3. Vision architecture

RSNA-winner-standard 2.5D; novelty budget goes to supervision, not architecture.

1. **Preprocessing (cached once to fp16 arrays on a Kaggle Dataset):** per series — sort slices by `ImagePositionPatient`·normal; fix MONOCHROME1/rescale; derive laterality from geometry; crop to fixed mm (≈140 mm) around image center → resize; percentile-normalize per series. Derive sequence type (plane × fluid-sensitive × fat-sat) from DICOM physics headers, not the buggy CSV.
2. **Series selection:** slot policy over available sequences — target slots: sag fluid-sat, sag non-fat-sat/T1, cor fluid, ax fluid (+PD where present); missing slots → learned null embedding. Start with 3–4 slots; RSNA 2024's 2nd place proves fewer well-chosen series can win.
3. **Backbone:** EfficientNetV2-S and ConvNeXt-T at 384² (thin-structure findings) and 256² (efficiency model); half the ensemble RadImageNet-init, half ImageNet-init (checked normalization contracts). No large ViTs — encoder scale is measured ≈ irrelevant here.
4. **Aggregation:** per-series adjacent-slice triplets (N≈16–32) → shared 2D encoder → BiGRU + attention-MIL over slices (max/top-2 pooling variants for focal findings: fracture, contusion) → cross-series attention over slot embeddings (+ PatientSex/age token — legitimate measured signal) → **12 sigmoid heads on a shared trunk** (per-finding heads only if the dashboard shows interference).
5. **Two-phase training (the 4,349+58 structure):** Phase A — full network on soft labels with confidence weights. Phase B — freeze encoder, retrain aggregator+heads on gold-58 (K-fold) + top-confidence pseudo-labeled studies (RSNA 2022 3rd-place pattern). Phase B is cheap → run per fold.
6. **Optional (+0.01–0.03 if compliance clears):** aux Dice segmentation head from SKM-TEA cartilage/meniscus masks on the last two encoder blocks.

## 4. Training recipe

- **Loss:** BCE on soft targets × per-sample-per-finding confidence weights; per-finding pos-weight from measured prevalence. No label smoothing (symmetric-noise tool; our noise is asymmetric).
- **Cleanup loop:** after first 5-fold round, confident learning on OOF preds to flag worst pseudo-labels → down-weight/relabel → retrain (proven RSNA 2024 2nd). One or two rounds; watch the gold anchor for over-cleaning.
- **Augs (MRI-appropriate):** flips with **laterality-aware label swap** (Med↔Lat pairs), small rotations/scale, intensity jitter, reverse slice order, per-study series mixing, manifold mixup on embeddings. No elastic warps that fake pathology.
- **Budget allocation by regime:** model-limited findings (PF OA, Lateral OA, Lat Meniscus) get resolution/slice-count experiments; teacher-limited (Baker's, Fracture, Contusion) get labeler iterations — GPU there is wasted.
- **Reproducibility:** seeded, config-logged (yaml per run), checkpoint-resume mandatory (Kaggle 12 h kernel cap makes it non-optional).

## 5. Inference pipeline

Budget (9 h = 32,400 s for ~1,300 studies ≈ 25 s/study ceiling — generous):

| Stage | Est. (main entry) |
|---|---|
| DICOM read + preprocess (parallel CPU workers) | ~1.5–2.5 h |
| 4–6 models × forward | ~1–2 h |
| Total | **~4 h** — half the cap, safe margin |

- **Efficiency entry (primary):** single EffNetV2-S (or smaller), 12–16 slices, 256², 3–4 slots, fp16 (TensorRT if stable in the rerun env), no TTA. Target **0.92+ AUC, <15 min**. At 0.01 AUC ≈ 720 s, accuracy dominates — all label-quality and per-finding tuning work lands here first.
- **Main-LB entry (bonus, derived):** rank-average of the efficiency line's 5 fold models + 1–2 diversity backbones (ConvNeXt-T / RadImageNet-init), per-finding member exclusion where measured (e.g. RadImageNet out of Baker's/Fracture). TTA: horizontal-flip-with-label-swap only if it survives the gold CI — likely dropped.
- Preprocessing code is shared verbatim between train cache and inference (one module) to kill train/test skew.

## 6. Risk register (top 5)

| # | Failure mode | Mitigation |
|---|---|---|
| 1 | **Overfitting decisions to gold-58** (thresholds, member selection, silence imputation all fitted on 58 points) | Bootstrap CIs on everything; decisions require CI-separated wins; permanent quarantine of gold from training; pre-committed final-selection rule (§2.5) |
| 2 | **Labeler degradation on non-English reports, unmeasurable** (gold may not cover all 12 languages) | Translate-then-label default; per-language hand-checks; confidence down-weighting; blend with independent public label sets |
| 3 | **Private-LB shakeup** (70% private, 0.03 separates 50 teams) | Local-CV-driven decisions; robustness-first final pick; diversity in ensemble; never chase public LB late |
| 4 | **Compute/logistics: 570 GB on free Kaggle quota** (~30 GPU-h/week, 12 h sessions, 20 GB output caps) | Preprocess once to fp16 cache stored as Kaggle Datasets (sliced by study shards); checkpoint-resume everywhere; small backbones (measured: scale doesn't pay); consider Colab/paid GPU only if quota binds after week 3 |
| 5 | **Compliance surprises** (hosted-LLM ruling, gated datasets, CC-BY-NC winner terms) | Re-verify #733965 + rules page immediately on joining; default stack uses only open-weights LLMs + clearly-open pretrained weights; gated data (MRNet/SKM-TEA) is optional add-on, never load-bearing |

## 7. Roadmap (ordered by expected LB impact per engineering hour)

| Window (2026) | Milestone | Exit criterion |
|---|---|---|
| **Aug 19–24** | Ops + ground truth: open network to Kaggle, rotate token → env secret; join competition; re-verify rules/#733965/GPU type on live pages; push **audit kernel** (ANALYSIS §2.3) → `audit.json` + dual-grouped folds | Audit numbers reproduced or corrected; folds frozen v1 |
| Aug 24–31 | **Baseline with correct CV:** preprocess cache v1; single EffNetV2-S, best public label set; 5-fold OOF + gold anchor + first 2–3 submissions to fit CV→LB slope | ≥0.90 public; slope measured; dashboard live |
| Sep 1–14 | **Label engine v1–v2** (§1): LLM triple-extraction, gold-calibrated thresholds, blend, confidence weights; retrain baseline per label version | Labels ≥0.90 vs gold (beats 0.893 public best); measurable LB gain from labels alone |
| Sep 8–21 (overlap) | **Architecture to full spec** (§3): multi-slot cross-series attention, two-phase training, laterality-swap augs; confident-learning cleanup round | Single model ≥0.925 public |
| Sep 21–Oct 5 | **Per-finding optimization:** model-limited resolution/slice experiments; teacher-limited labeler iterations; silence imputation; pooling variants | Per-finding AUC gains with CI-separated evidence |
| Oct 1–15 | **Ensemble + efficiency entry:** 4–6 member rank ensemble; distill/prune efficiency model, fp16/TRT, runtime harness | Ensemble ≥0.935 public; efficiency model 0.92+/<15 min; **entry deadline Oct 15** |
| Oct 15–22 | Freeze: robustness checks (fold re-splits, seed variance), final 2 selections per §2.5, submission dry-runs against 9 h harness | Two selected subs, both rerun-verified |

**Cross-cutting rule:** any experiment that can't show a CI-separated win on the two-tier validation within its window is cut — the schedule has zero slack for architecture tourism.

---

## The 3 highest-leverage actions to start immediately

1. **Open the pipe and freeze the ground truth (today):** allow kaggle.com in the environment network policy, store the (rotated) `KAGGLE_API_TOKEN` as a secret, join the competition, re-verify on the live pages the three load-bearing unverified facts — efficiency formula parenthesization, hosted-LLM ruling #733965, rerun GPU type — then push the audit kernel and freeze dual-grouped folds v1.
2. **Stand up the two-tier validation harness before any model exists:** fold CSV + OOF/gold-anchor evaluator with bootstrap CIs + per-finding dashboard. Every subsequent decision routes through it; built once, it prevents the failure mode that kills most teams here.
3. **Ship label engine v1 against the 0.893 public ceiling:** LLM triple-extraction prompt with the gold rubric embedded, translate-then-label, thresholds fitted by K-fold on gold-58. Beating the best public label set is the single largest expected LB gain available and gates everything downstream.
