# ANALYSIS — RSNA Knee Abnormality Detection (Kaggle 2026)

Phase 1 (problem ingestion) + Phase 2 (code audit) findings. Compiled 2026-08-19.

**Sourcing caveat:** kaggle.com and rsna.org were unreachable from this environment (egress proxy blocks them). Facts below were reconstructed from (a) search-result snippets of official pages and (b) verbatim transcriptions of the Kaggle Overview/Data/Rules/Prizes pages committed to public competitor GitHub repos. Confidence tiers:

- **[A]** near-primary: verbatim competition-page transcriptions (AhmadEnan, DsChauhan08, ahmed-hesham07, sh0ch repos)
- **[B]** competitor field notes measured against the real data (dk2lone, stephengardnerd, Soham-47 repos)
- **[C]** press (RSNA news, AuntMinnie — snippet-level)
- **[U]** unverified — re-check on the live pages before committing engineering time

---

## 1. Competition spec

### 1.1 Task

Predict **12 independent binary findings per knee MRI study** (`StudyInstanceUID`), output as probabilities in [0,1]. Multilabel, not multiclass. **[A]**

CSV column headers (note the apostrophe in `Baker's` — a repeatedly reported string-handling footgun):
`ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`

### 1.2 Gold label definitions (severity-thresholded, specificity-biased) **[B, verbatim]**

Test ground truth is **image-derived** by two MSK subspecialty radiologists + a third adjudicator, with an explicit rule that borderline / "on the fence" cases are graded **negative**:

| Finding | Positive iff | Domain note (what/where to look) |
|---|---|---|
| ACL | High-grade partial or full tear: discontinuity or >50% fibres disrupted. Mucoid degeneration/thickening without discontinuity → negative | Best on sagittal fluid-sensitive (PD/T2 fat-sat); secondary coronal |
| MCL | High-grade partial or complete **acute** tear with edema. Low-grade sprain/chronic → negative | Coronal fluid-sensitive; medial joint line |
| Medial / Lateral Meniscus | Signal contacting the articular **surface on ≥2 images**, or truncated/diminutive/displaced fragment. Intrasubstance degeneration → negative | Sagittal + coronal PD (non-fat-sat helps); "≥2 images" is why slice count matters |
| Medial / Lateral / PF OA | **≥1 cm area of >50%-thickness cartilage loss** in that compartment | Cartilage-sensitive sequences; PF = axial, tibiofemoral = coronal/sagittal |
| Effusion | **Moderate or large** joint fluid (trace/small/mild → negative) | Axial/sagittal T2; suprapatellar pouch |
| Synovitis | Synovial lining inflammation and thickening | Hard without contrast; often co-occurs with effusion |
| Baker's | **Moderate or large** popliteal cyst (small → negative) | Axial/sagittal T2; posteromedial, gastrocnemius-semimembranosus interval |
| Contusion | Marrow edema-like signal from impact **without** discrete fracture line | Fluid-sensitive fat-sat; bone marrow |
| Fracture | Acute cortical break / fracture line | T1 + fluid-sensitive; cortex |

**Structural consequence:** a careful human reading only the report agrees with gold labels at ~**82.5%** **[B]**. Report language does not encode these severity thresholds (a report saying "small effusion" is a gold NEGATIVE). The label function, not the backbone, is the dominant error term.

### 1.3 Metric — CORRECTION TO OUR BRIEF

**Plain unweighted macro-averaged ROC-AUC over the 12 findings**: `Score = (1/12) Σ AUCᵢ`. **[A, three independent transcriptions]**
There is **no weighting**. Only rank order matters → rank-averaging beats probability-averaging for ensembles; probability calibration is irrelevant to the score (but matters for our own diagnostics).

### 1.4 Format & constraints

| Item | Value | Conf. |
|---|---|---|
| Type | Code competition, notebook rerun on hidden test | [A] |
| Submission | `submission.csv`: `StudyInstanceUID` + 12 float columns | [A] |
| Runtime | ≤ **9 h** (32,400 s), CPU or GPU | [A] |
| Internet | **Disabled at inference** | [A] |
| Daily submissions | 5/day, 2 final selections | [A] |
| External data | Allowed if freely & equally available at no cost | [A] |
| Pretrained models | Allowed (public). DINOv2/v3, RadImageNet, BiomedCLIP in common use via Kaggle Models | [A/B] |
| Hosted LLM APIs for report labeling | **Reportedly allowed by host ruling** (discussion #733965). One secondary source contradicts this. **Re-verify on the live thread before depending on it** | [B/U] |
| GPU type in rerun env | **Unverified.** Competitor logs reference T4×2; P100/L4 unconfirmed | [U] |
| Winner obligations | Code + weights + methodology doc + video, CC-BY-NC 4.0 | [A] |

### 1.5 Efficiency prize track

`EfficiencyScore = AUC / (Benchmark − maxAUC) + RuntimeSeconds / 32400` — **lower is better**. **[A, two matching transcriptions]**

- `Benchmark` = sample_submission private-LB score; `maxAUC` = best private-LB AUC → denominator negative → higher AUC drives score more negative; runtime adds penalty.
- Exchange rate ≈ **0.01 AUC ≈ 720 s runtime** [B].
- Eligibility: must be one of your 2 selected submissions AND beat sample_submission on private LB.
- Practical accuracy floor for placing ≈ **0.915 AUC** [B]. Prizes $7k/$6k/$5k. Host-run daily efficiency LB (rank only during training period).
- Prize pool overall: **$77k** — main LB $59k across 10 places ($9k 1st), efficiency $18k across 3.

### 1.6 Timeline **[A/C, unanimous]**

| Milestone | Date (2026) |
|---|---|
| Launch | Jul 30 |
| Entry + team-merger deadline | **Oct 15, 23:59 UTC** |
| Final submission | **Oct 22** |
| Winner obligations due | Nov 5 |

Today (Aug 19) → **57 days to entry, 64 to final.**

---

## 2. Data profile

### 2.1 Verified facts

| Fact | Value | Conf. |
|---|---|---|
| Training studies | **4,407** | [A/B] |
| … with gold labels | **58** (1.3%), all-or-nothing (labeled study has all 12) | [A/B] |
| … with free-text report only | **4,349** | [A/B] |
| Training series | **24,371** (sag 9,864 / cor 8,609 / ax 5,898) | [B measured] |
| DICOM files / size | ~819k files, ≈570 GB compressed (1.1–1.6 TB decompressed) | [A/B] |
| Layout | `{train,test}_series/{StudyUID}/{SeriesUID}/*.dcm`, ~20–45 slices/series | [A] |
| Series metadata | `train_series.csv`: `anatomical_plane`, `fluid_sensitive`, `fat_suppression` | [A] |
| Test set | ~**1,300 studies**, **no reports provided** | [A] |
| Public/private split | **30% / 70%** | [B] |
| Reports | 12 languages, 16 sites, 5 continents (Belgium, Argentina, Thailand, Taiwan, Turkey, Canada named) | [A/C] |

**The pivotal design fact: reports exist only at training time.** The model that runs at inference sees pixels only. Reports are a supervision channel, not an input.

### 2.2 Measured data quirks (community, [B])

1. **Slice order from filenames is wrong ~95% of the time.** Sort by `ImagePositionPatient` projected onto the `ImageOrientationPatient` slice normal.
2. **FOV varies hugely** (70+ distinct values, median ~160 mm). Crop to fixed **millimetres** then resize (e.g. 130 mm→336 px ≈ 0.39 mm/px); pixel-count crops erase 1–3 mm pathology.
3. **Laterality must be derived from imaging geometry**, not assumed — 5/12 labels are medial/lateral-specific and a flipped knee swaps them.
4. `fluid_sensitive` and `fat_suppression` columns are **byte-identical across all 24,371 rows** — one is redundant/buggy; derive sequence type from DICOM headers (TR/TE, SequenceName, fat-sat flags) instead.
5. **Uneven slot availability:** sagittal T1 in only ~42.5% of studies; non-fat-sat sequences in ~57% of test. Any fixed "6-slot" input needs a missing-slot policy.
6. **49 duplicate-report groups covering 183 studies (4.2%)** — largest is 37 byte-identical Turkish normal-knee templates. These leak across naive CV splits.
7. **Gold-58 is enriched, not representative:** zero all-negative studies, mean 4.14 positive findings/study. Never use it as a prevalence-representative holdout.
8. **Site/scanner signal exists but is weak:** metadata-only classifier ~0.65 AUC under random folds → ~0.50–0.60 site-grouped; vision models lose only ~0.003 under scanner-grouped folds. No metadata shortcut; still group folds.
9. **CV→LB relation is a slope, not an offset:** fitted `LB ≈ 0.825 × gold-CV + 0.184` [B]. Improvements measured on gold shrink ~17% on the board.
10. `PatientSex` is legitimate signal (ACL: M 54% vs F 32% positive; Medial OA: F 45% vs M 12%).

### 2.3 Data audit still to run (Kaggle-side)

The dataset never leaves Kaggle (573 GB; user directive: all data/GPU work runs on Kaggle notebooks via API-pushed kernels). First kernel to push once network to kaggle.com is opened from this environment (`KAGGLE_API_TOKEN` as secret env var — never committed):

- Recompute §2.1/§2.2 numbers from scratch (trust nothing above blindly): study/series/instance counts, slices-per-series distribution, plane/sequence availability matrix per site, pixel spacing/FOV histograms, manufacturer/field-strength distribution.
- Report language detection (fasttext/langid on `train.csv` report text) → language × site crosstab.
- Gold-58: per-finding prevalence table + co-occurrence matrix.
- Duplicate-report union-find groups; scanner (`DeviceSerialNumber`/`StationName`) groups → persist the **dual-grouped fold assignment** as a versioned CSV.
- DICOM landmines: orientation flips, `PhotometricInterpretation=MONOCHROME1`, `RescaleSlope/Intercept`, bit depth, localizer series to exclude.
- Output: single `audit.json` + fold CSV pulled back via `kaggle kernels output`.

---

## 3. Community intel (all [B]; dates noted)

### 3.1 Leaderboard landscape (mid-Aug 2026)

- ~1,100–1,840 teams. **Top public ≈ 0.945**; 10th ≈ 0.938; positions 1–49 span 0.033.
- Best public notebook **0.920** (Tony Li, DINO+RadImageNet rank ensemble); widely-copied baseline 0.899 sits ~rank 60–80. Serious teams run 1.5–3 AUC points ahead of anything public.
- 70% private + extreme compression → **high shakeup risk**; small public-LB moves are noise. Trustworthy local validation is the game.

### 3.2 Notebook / label-set ecosystem

Key public assets: Pilkwang Kim's multilingual extractor + 6-slot DINOv2 baseline (0.891); prvsiyan's negation-aware extractor (0.906); Mattia Angeli DINOv3 collab (0.917); Tony Li rank ensemble (0.920); wguesdon's rigorous resolution study (0.815 but best methodology). Most-reused label datasets: Pilkwang Kim LLM labels (94 notebooks), stevenleehans LLM labels v4_blend (**0.8927 macro-AUC vs gold** — best public), Barun Kumar stratified folds + soft labels.

Caution: vote counts ≠ quality; several high-scoring notebooks show `Accelerator: None` / empty logs — the visible code never produced the claimed score. Check Logs tab before forking anything.

### 3.3 Converged findings (what actually moves the score)

1. **Label quality > architecture.** LLM labels reach 0.878–0.893 AUC vs gold; lexicon-only caps at 0.814. Weakest columns: Synovitis ~0.79, Fracture ~0.79.
2. **Encoder scale ≈ irrelevant** (DINOv2-base vs -small: +0.001). **Slice count matters more**: 6→12 slices = +0.0056.
3. **Per-finding regime split:** PF OA, Lateral OA, Lateral Meniscus are **model-limited** (better vision helps); Baker's, Fracture, Contusion are **teacher-limited** (better labels help). Allocate effort accordingly — almost nobody does.
4. **"Not addressed" in a report is finding-specific signal** (silence → Baker's 3% positive but Synovitis 34%). Targeted imputation (e.g. fill undecided synovitis from effusion) moved that column 0.678→0.790.
5. **Rank-average ensembling; max/top-2 pooling** over slices for focal findings (fracture, contusion). RadImageNet independently excluded from Baker's + fracture columns by multiple teams.
6. **Single well-tuned model reaches 0.915–0.934**; giant ensembles overrated; OOF≈LB for well-tuned singles.
7. Checkpoint **normalization contracts** (BiomedCLIP σ ≈ 1.2× ImageNet) silently corrupt runs when hardcoded.

### 3.4 Gaps nobody is exploiting

- **Efficiency track under-contested:** floor ≈0.915 AUC; a competitor's full 5-model entry ran in ~10 min — runtime is easy, accuracy is the binding constraint. A genuine 0.92+ single model under ~15 min is arguably the highest-EV play for $18k.
- Per-finding resolution/coverage tuning; decorrelated label-set blending (the weaker extractor adds the most lift); model- vs teacher-limited budget allocation.

### 3.5 Open compliance questions **[U]**

Whether gated-but-free datasets (MRNet, fastMRI+, OAI, SKM-TEA) satisfy "equally accessible at no cost", and whether research-use-only weights conflict with CC-BY-NC winner obligations. Not clarified by host. Also re-verify the hosted-LLM ruling (#733965) on the live thread.

---

## 4. Literature synthesis → design implications

### 4.1 Knee MRI classification

- **MRNet** (Bien 2018): per-slice AlexNet features → max-pool over slices → per-view model, logistic-regression fusion of 3 views. AUC: ACL 0.965, meniscus 0.847. The task-shape template.
- **ELNet** (2020): 0.2M params, multi-slice norm + BlurPool, beats MRNet on meniscus (0.904). Existence proof that tiny nets suffice → efficiency track.
- **CoPAS** (Nature Comms 2024): the closest published analogue — **same 12-finding knee task**, cross-plane attention, mean AUC **0.812** with clean labels. Implication: the LB above 0.90 is won by supervision quality + ensembling, not novel attention; and the per-class difficulty spread (ACL ~0.96 vs subtle findings ≪) means **rare/subtle classes decide a macro metric**.
- External data: **RadImageNet** pretraining gave **+4.5–4.8 AUC pts on exactly ACL/meniscus MRI** in small-data regimes — most relevant pretraining evidence. MRNet dataset (1,370 exams) as pseudo-label/pretraining source; SKM-TEA (cartilage/meniscus masks) for an auxiliary segmentation head. All pending §3.5 compliance.

### 4.2 Report weak-labeling

- Progression CheXpert (rule, F1 0.743) → CheXbert (BERT distilled from rules + ~1k expert annotations, 0.798) → GPT-4-class LLMs (~0.90). Biggest LLM gains on rare/ambiguous classes — exactly our weak columns.
- **CheXbert's structure maps 1:1 onto our 4,349+58 split**: train a labeler on noisy signal, calibrate/fine-tune on the small gold set.
- Open-weight LLMs (Mixtral/Qwen-class) demonstrably sufficient for report labeling (Radiology 2025; Academic Radiology 2025) — no API dependence required.
- Multilingual: zero-shot LLM labeling measurably degrades off-English → **translate-then-label** with one strong English labeler is the robust default; A/B against direct multilingual prompting on gold + hand-checked samples. Turkish post-posed negation ("efüzyon izlenmedi") breaks English-pattern extractors.
- Noisy-label toolbox, in order of proven value here: soft labels + confidence weights → **confident learning** (used by RSNA 2024 2nd place) → co-teaching/DivideMix/BoMD only if needed. Label smoothing is symmetric-noise-shaped — wrong for our asymmetric (report-optimistic vs gold-strict) noise; prefer per-finding noise-aware soft targets.
- ConVIRT/CheXzero-style image–text contrastive pretraining is a label-free way to use all 4,349 pairs, but at 1/85th of CheXzero's scale — secondary pretraining stage at best.

### 4.3 Architecture under 9 h

Across RSNA 2022/2023/2024 winners the recipe is stable: **(optional localizer →) 2.5D slice triplets → 2D CNN (EffNetV2-S / ConvNeXt-T / ResNeSt50) → BiLSTM/GRU + attention-MIL pooling → study head**. Full 3D wins only for dense localization tasks (RSNA 2025 aneurysm) and needs multi-A100-days. Video models: no evidence, skip.

### 4.4 RSNA winners' playbook — transferable pieces

| Source | Piece we reuse |
|---|---|
| RSNA 2022 3rd (darraghdog) | **Freeze slice encoder, train only aggregator on scarce study-level labels** — exactly our weak-4,349 / gold-58 structure; `F.interpolate` for variable-length series |
| RSNA 2023 1st (Nischaydnk) | `(32×3, 384²)` triplets → CNN → GRU → max-pool logits; **soft labels = study label × per-slice structure visibility**; aux Dice segmentation heads **+0.01–0.03** |
| RSNA 2024 2nd (brendanartley/ianpan) | **Dropped an entire plane and still took 2nd** (fewer, better-chosen series); confident learning on OOF; equal-weight pseudo-labeling; augs: reverse slice order, side-mixing, manifold mixup; variable slice-count/resolution members for ensemble diversity |
| RSNA 2025 2nd | Modality/sequence-specific auxiliary heads for heterogeneous multi-site input |

---

## 5. Code audit (Phase 2)

**The repository is empty** — zero commits, zero files before this document. Verdicts:

- Leakage risks: none exist yet (nothing to leak). The leakage-prone decisions are all ahead of us and are pinned in STRATEGY.md §2: dual-grouped study-level folds, labeler prompts/thresholds fitted only on in-fold gold data, fold assignment versioned as data.
- Silent correctness bugs / inefficiency / dead code: vacuously none.
- Ground rules from here (non-negotiable, from the brief): every line justified; study-level grouped CV only; cache preprocessed arrays once; fixed seeds + logged configs + resumable training; measure before choosing.

## 6. Environment & operations facts

- This Claude Code remote environment currently **cannot reach kaggle.com** (proxy CONNECT 403). Fix: allow `kaggle.com`, `www.kaggle.com`, `storage.googleapis.com` in the environment's network policy (or unrestricted), set `KAGGLE_API_TOKEN` as an environment secret. The token pasted in chat should be **rotated** and stored only as a secret.
- Operating model (user directive): **all data/GPU work on Kaggle** — author code in this repo, `kaggle kernels push` with GPU + competition data source, pull back only small artifacts (audit JSON, OOF CSVs, weights). The 570 GB dataset is never downloaded locally. Kaggle free GPU quota (~30 h/week, T4/P100-class) is the training budget; heavy runs must be sliced into ≤12 h kernel sessions with checkpoint-resume.
