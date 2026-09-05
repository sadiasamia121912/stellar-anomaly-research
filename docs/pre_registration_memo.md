# Pre-Registration Memo
## Beyond Detection: Morphological Fingerprinting and Contrastive Characterization of Novel Stellar Anomalies in TESS Data

**Author:** Samin | Albukhary International University (AIU)
**Date:** [fill in]
**Status:** Locked before any test-split result is generated. Per Part H (Step 2) and the Part I gate, this document must be complete before Step 6 (ground truth construction) or before Step 3's dataset construction is treated as final.

---

## 1. Corrected Statistical Framing (locking prior audit corrections)

1.1 **Anomaly-Novelty correlation:** The pilot r=0.9617 statistic partially reflects that the novelty score formula includes the VAE reconstruction-error term. Before this correlation is reported again, it will be recomputed with the shared VAE term removed from one side. Only the corrected value is reportable evidence.

1.2 **Explanation-generation mechanism:** The contrastive explanation engine is template-based feature attribution (fixed sentence template filled with the feature of maximum difference + similarity percentage), not free-form natural language generation (no LLM). This is stated explicitly in the Methodology section, not implied.

1.3 **Discovery-framing language:** Even in the best-case scenario, the paper will never use "new class" or "discovery" language. The maximum claim language is: "a statistically robust, previously uncatalogued photometric behavior warranting spectroscopic follow-up."

1.4 **Journal-quartile target:** MNRAS is the confirmed Q1 target. Astronomy & Computing is a strong, legitimate Q2, Scopus/SCIE-indexed first-submission target -- not itself a Q1 outcome.

## 2. Two-Track Primary Metrics (R5)

**Track A -- Anomaly Detection**
- Primary metric: AUC-PR
- Secondary: precision/recall at the pre-registered operating point (frozen on validation, Step 6)
- Evaluated against: injection-recovery ground truth
- At a predefined, reported injection prevalence (Section 5) -- explicitly NOT the natural anomaly rate

**Track B -- Novelty Detection**
- Primary metric: leave-one-class-out recovery rate
- Also reported: known-class false-novel rate, novelty-score distributions, ranking metrics

**Descriptive/exploratory only (never used for a significance claim):** silhouette score, cluster purity, similarity percentages.

**Hard rule:** Track A results are never used, alone, as evidence for a Track B (novelty) claim. They answer different questions.

## 3. Seed Schedule (R6)

- Standard experiments: >=5 seeds
- Headline comparisons: >=10 seeds -- specifically:
  - Model C vs. Model D
  - Proposed pipeline vs. External Baseline 1 (Villar-style)
  - Proposed pipeline vs. External Baseline 2 (Webb-style)
  - Proposed pipeline vs. External Baseline 3 (CLARA-style)
- Seed count increased further if model rankings are unstable across seeds.
- Seeds quantify training/algorithmic stochasticity only -- they are not treated as independent observational units for population-level claims (see Section 4).

## 4. Statistical Protocol (R7)

Three distinct sources of uncertainty are kept separate and never pooled:

1. **Training stochasticity** -- seed-level variability (Section 3). Reported as secondary stability evidence only.
2. **Population uncertainty** -- star-level bootstrap on the test set. The physical star (TIC ID) is the independent observational unit, never the cadence. Where a star contributes multiple injected instances, bootstrap resampling draws whole stars as blocks -- all of a resampled star's instances travel together, never resampled independently. A resample with too few positive instances to compute a stable AUC-PR is discarded, and the discard rate is reported alongside the CI.
3. **Model comparison** -- primary inferential basis is the paired difference in test-star-level predictions between models, with 95% CI via paired star-level bootstrap and a pre-specified permutation procedure.

**Seed x bootstrap combination order:** for any multi-seed comparison, per-star predictions are averaged across the full seed set first; star-level paired bootstrap then resamples stars from this seed-averaged set. Seed-level spread is reported separately as a stability diagnostic, never folded into the primary CI.

**Test selection:** no dependence on a normality test at low seed counts -- use paired bootstrap, Wilcoxon signed-rank, or another pre-specified robust procedure. Report an effect size (Cohen's d or rank-biserial correlation) alongside every test.

**Multiple-comparisons correction:** Benjamini-Hochberg FDR, applied within three closed, exhaustive, pre-registered families (assigned now, before any test-split result is viewed):

- **Family 1 (Track A confirmatory):** the four-model ablation comparisons (A vs B, B vs C, C vs D) + every proposed-vs-baseline comparison (vs. Baselines 1, 2, 3) + the anomaly-rate permutation test.
- **Family 2 (Track B confirmatory):** leave-one-class-out recovery-rate and known-class false-novel-rate tests, per held-out class.
- **Family 3 (exploratory/robustness):** the faithfulness check + the TESS-systematics covariate-adjusted tests.

**Explicitly excluded from all three families** (not hypothesis tests): the hyperparameter sensitivity sweep (descriptive) and the novelty-formula selection (internal, validation-only engineering choice -- see Section 5).

## 5. Injection Prevalence & Class-Imbalance Strategy

**Order of operations (fixed):** preprocessing/normalization is fit on train's clean, non-injected curves only; synthetic anomalies are injected afterward, into the normalized flux.

**Injection prevalence:** 50% injected / 50% clean confirmed-normal (1:1 ratio), applied uniformly within each evaluation set (train, validation, test) across every injection-strength bin. This ratio is fixed and documented per split before any test-split number is generated. Chosen for stable precision/recall/AUC-PR estimation across injection-strength bins -- explicitly distinguished from, and not intended to represent, the pipeline's natural anomaly rate.

**Parameter ranges:** documented separately for train/val/test (transit depths 0.1-5%, flare amplitudes, injected periodic signals, signal locations, noise/background variation). The test split includes at least one held-out parameter regime or morphology combination not present in train/validation injections, to test generalization rather than interpolation.

**Class-imbalance handling (training only):** Class-weighted loss, applied uniformly across the CNN fingerprint model, the ablation classifiers, and all three external baselines under R8's equal-treatment rule. Chosen over oversampling to avoid duplicating real light-curve instances (overfitting risk on a limited real-data sample), and over balanced mini-batch sampling for simpler, more transparent reproducibility. Applied only within the training split; validation/test prevalence is left untouched.

**Thresholds (two, kept distinct, both frozen on validation only, before test is touched):**
- Anomaly-detection operating-point threshold (Step 6)
- Novelty threshold (Step 10)

## 6. Reference-Library Specificity-Testing Method (R12)

**Chosen method: k-fold rotation within each class's exemplar set** (the pre-registered default).

For every reference class remaining in the library during a leave-one-class-out fold: build that class's reference distribution on k-1 folds of its own exemplars, test the false-novel rate on the held-out fold, and rotate until every exemplar has served as a held-out test point exactly once.

**Hard rule:** specificity is never tested on the same exemplars used to build that class's own reference distribution -- this would be circular and would trivially understate the false-novel rate.

This method is used uniformly across all reference classes -- never mixed ad hoc with the independent-held-out-stars alternative.

**Exemplar-count rule (M6):** reference classes intended for quantitative/headline leave-one-class-out claims must reach >=15-20 independent exemplars. Classes below this bar are retained if scientifically useful, but reported as exploratory only, with wide confidence intervals -- never as headline evidence.

---

## Non-negotiable rules carried over from Part A (referenced, not re-derived here)

- **R1/R1b:** Star-level (TIC ID) train/val/test split. S2 and S8 are deterministically pre-assigned to TEST before the general random split is generated -- never revisited based on any result.
- **R2:** The reference library is fixed, external, and fully disjoint from the train/val/test pool.
- **R9:** S2 and S8 are never used to tune any hyperparameter, threshold, or feature set. All tuning happens on validation only; S2/S8 are evaluated under an independently frozen configuration and reported as a case study.
- **R10:** Every reported number traces to a fixed seed, a pinned environment, full data provenance (TIC ID, sector, cadence, product, pipeline version), a logged config file, and a tagged repo commit.
- **R11:** Anomaly/novelty scores are checked against known TESS systematics (momentum dumps, scattered light, sector-stitch boundaries) via rank correlation, stratified comparison, and covariate-adjusted regression -- not a bare linear correlation.

---

## Sign-off

This memo is treated as locked and unchanging once dated below, except by an explicitly logged amendment (with reason) made **before** the relevant test-split result is generated. No revision is permitted after a result related to that decision has been seen.

**Date locked:** ______________
**Locked by:** Samin


## R12 specificity scheme (corrected)

5-fold rotation applies uniformly to every reference-library class, regardless of exemplar count. An earlier draft applied independent held-out splits to large classes and k-fold to small ones — this was reverted because k-fold is valid at any sample size, so uniform application removes the ad hoc per-class mixing without any statistical cost.

## Hand-crafted feature normalization scale (amendment, 2026-09-06)

The 13 hand-crafted candidate features (`src/features.py`) are computed on a per-curve, min-max-scaled-to-[0,1] flux, uniformly interpolated to 1000 points (matching the original pilot's design) — **not** the VAE's global z-score normalization (`GLOBAL_NORM_MEAN`/`GLOBAL_NORM_STD`). This is necessary, not optional: `Dim_Frac`/`Bright_Frac`'s fixed 0.3 threshold, and several other features' effective scale, are only meaningful against a curve bounded in [0, 1]; on z-scored flux they would silently measure something else. This is a model-specific representation choice, downstream of and consistent with R3's one common pipeline (detrending, gap-handling, outlier-clipping, which happens once, upstream, identically for every model) — exactly analogous to the VAE having its own binning/windowing step. Applied identically to every star at train/val/test time. Logged here before the frozen feature set (below) is used to train Model B.

## Frozen hand-crafted feature set (amendment, 2026-09-06)

Decided per M3: candidate sets compared on VALIDATION only (never test), using the pre-defined, closed list {all 13, drop the 4 weakest by train-split Track A importance, drop the 6 weakest}. Track A alone was **not** used as sufficient grounds to drop a feature — a second check, whether each candidate weak-for-Track-A feature (`Dom_Freq`, `Rise_Fall_Ratio`, `N_Peaks`, `N_Troughs`) actually carries no signal for Track B's reference-class separation either, was required before any feature could be cut (a feature that only matters for periodicity-defined classes — pulsators, rotational variables, eclipsing binaries — could look useless against synthetic injection recovery while still being essential to the novelty/reference-library comparison; M4's Track A/B separation rule forbids using Track A evidence alone for a Track B claim, and symmetrically it should not be used alone to cut something Track B needs). See `results/feature_importance/frozen_feature_set.json` for the full validation numbers, the Track B permutation-importance table, and the final decision with reasoning. This decision was made by Claude on the user's explicit delegation (chat, 2026-09-06); reversible if the full-scale GPU run's real Model B/C/D results contradict it.
